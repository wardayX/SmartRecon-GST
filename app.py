import os
import io
import re
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd
import pdfplumber
from fuzzywuzzy import process as fuzzy_process
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util as sbert_util
from transformers import pipeline as hf_pipeline



try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
    print("INFO: pytesseract found.")
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("CRITICAL WARNING: pytesseract library not found. OCR fallback for DocVQA will likely fail or perform poorly.")
    print("Install with: pip install pytesseract")
    print("AND ensure Tesseract OCR engine is installed and in your system PATH (https://github.com/UB-Mannheim/tesseract/wiki).")
except Exception as e:
    PYTESSERACT_AVAILABLE = False
    print(f"CRITICAL WARNING: Error initializing pytesseract: {e}. OCR fallback for DocVQA may fail.")
    print("Ensure Tesseract OCR engine is installed and in your system PATH.")

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF2IMAGE_AVAILABLE = True
    print("INFO: pdf2image found.")
    try:
        pass
        print("INFO: Poppler (dependency for pdf2image) seems to be accessible.")
    except Exception as e:
        print(f"WARNING: Poppler (dependency for pdf2image) might not be correctly installed or in PATH: {e}")
        print("PDF processing will likely fail. Install Poppler utilities for your OS.")

except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("CRITICAL WARNING: pdf2image library not found. PDF processing for OCR will fail.")
    print("Install with: pip install pdf2image")
    print("AND ensure Poppler is installed on your system (e.g., sudo apt-get install poppler-utils or conda install -c conda-forge poppler).")


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
HSN_PDF_PATH = r"HSN-Codes-for-GST-Enrolment.pdf"
GST_CSV_PATH = r"cbic_gst_goods_rates_exact.csv"
SENTENCE_MODEL_NAME_GST_FINDER = 'all-mpnet-base-v2'
SENTENCE_MODEL_NAME_ITEM_MATCHER = 'all-MiniLM-L6-v2'
QA_MODEL_NAME_GST_FINDER = 'deepset/roberta-base-squad2'
DOC_QA_MODEL_NAME = "impira/layoutlm-document-qa"


df_merged_gst_finder = pd.DataFrame()
sbert_model_gst_finder = None
corpus_embeddings_gst_finder = None
qa_pipeline_gst_finder = None
all_descriptions_for_fuzzy_gst_finder = []
doc_qa_pipeline = None
sbert_model_item_matcher = None


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_very_secret_key_please_change_this_for_production'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

QUESTIONS_FOR_OCR = {
    "invoice_number": "What is the invoice number or bill number?",
    "invoice_date": "What is the invoice date or bill date?",
    "supplier_name": "What is the supplier's name or company from whom the goods/services are sold?",
    "supplier_address": "What is the supplier's full address?",
    "supplier_gstin": "What is the supplier's GSTIN or VAT registration number?",
    "buyer_name": "What is the buyer's name or customer to whom the goods/services are sold?",
    "buyer_address": "What is the buyer's full address or delivery address?",
    "buyer_gstin": "What is the buyer's GSTIN or VAT registration number?",
    "subtotal": "What is the subtotal amount or total amount before any taxes?",
    "total_cgst": "What is the total CGST amount?",
    "total_sgst": "What is the total SGST or UTGST amount?",
    "total_igst": "What is the total IGST amount?",
    "total_gst": "What is the total GST amount (sum of CGST, SGST, IGST)?",
    "grand_total": "What is the grand total amount, net payable, or final bill amount?",
    "items_detail_raw": "List all line items detailing description, quantity, rate per unit, and total amount for each item."
}


class GSTReconciliationEngine:
    def __init__(self, tolerance=0.01, sbert_item_match_threshold=0.65):
        self.tolerance = tolerance
        self.sbert_item_match_threshold = sbert_item_match_threshold
        global sbert_model_item_matcher
        self.sbert_matcher = sbert_model_item_matcher

    def _parse_float(self, value, default=0.0):
        if value is None: return default
        try:
            cleaned_value = str(value)
            cleaned_value = re.sub(r'[₹$€£,]', '', cleaned_value)
            cleaned_value = cleaned_value.strip()
            if re.search(r'\d+\.\d{3},\d{2}$', cleaned_value):
                cleaned_value = cleaned_value.replace('.', '', cleaned_value.count('.') -1).replace(',', '.')
            elif re.search(r'\d+,\d{3}\.\d{2}$', cleaned_value):
                cleaned_value = cleaned_value.replace(',', '')
            cleaned_value = re.sub(r'[^\d\.\-]', '', cleaned_value)
            if cleaned_value.count('.') > 1:
                 parts = cleaned_value.split('.')
                 cleaned_value = parts[0] + '.' + "".join(parts[1:])
            return float(cleaned_value) if cleaned_value and cleaned_value not in ['-', '.'] else default
        except (ValueError, TypeError):
            return default

    def validate_gstin(self, gstin_raw):
        if not gstin_raw or not isinstance(gstin_raw, str): return False, gstin_raw
        gstin = re.sub(r'[^A-Z0-9]', '', str(gstin_raw).strip().upper())
        if len(gstin) != 15: return False, gstin
        pattern = r'^[0-9]{2}[A-Z0-9]{10}[0-9][A-Z][0-9A-Z]$'
        return bool(re.match(pattern, gstin)), gstin


    def map_ocr_output_to_reconciler_format(self, ocr_output):
        if not isinstance(ocr_output, dict): return {'items': []} 
        if ocr_output.get("error"):
            print(f"Warning: OCR output contains an error: {ocr_output['error']}")
            mapped_with_error = {key: f"OCR Error: {ocr_output['error']}" for key in QUESTIONS_FOR_OCR}
            mapped_with_error['items'] = []
            mapped_with_error['ocr_main_error'] = ocr_output['error']
            return mapped_with_error

        mapped = {}
        for key_q, _ in QUESTIONS_FOR_OCR.items():
            key_m = key_q.replace('_raw', '') 
            raw_value = ocr_output.get(key_q) 

            if isinstance(raw_value, list) and raw_value:
                if key_q == "items_detail_raw":
                    mapped[key_m] = "\n".join(str(v) for v in raw_value if v).strip()
                else:
                    mapped[key_m] = str(raw_value[0]).strip() if raw_value[0] else None
            elif isinstance(raw_value, str):
                mapped[key_m] = raw_value.strip()
            else:
                mapped[key_m] = raw_value

        if mapped.get('total_gst') is None or self._parse_float(mapped.get('total_gst')) == 0:
            cgst = self._parse_float(mapped.get('total_cgst', 0))
            sgst = self._parse_float(mapped.get('total_sgst', 0))
            igst = self._parse_float(mapped.get('total_igst', 0))
            if cgst > 0 or sgst > 0 or igst > 0:
                mapped['total_gst'] = cgst + sgst + igst

        for amt_key in ['subtotal', 'total_cgst', 'total_sgst', 'total_igst', 'total_gst', 'grand_total']:
            mapped[amt_key] = self._parse_float(mapped.get(amt_key))

        mapped_items = []
        line_items_text_blob = mapped.get('items_detail')
        
        # *** Initialize current_item_desc_parts here ***
        current_item_desc_parts = [] 

        if line_items_text_blob and isinstance(line_items_text_blob, str):
            item_lines = line_items_text_blob.strip().split('\n')
            for line_text in item_lines:
                line_text = line_text.strip()
                if not line_text or len(line_text) < 3: continue
                numbers_on_line = re.findall(r'[₹$€£]?\s*(\d[\d,\.]*\d|\d+)\s*[₹$€£]?', line_text)
                parsed_numbers = [self._parse_float(n) for n in numbers_on_line]
                parsed_numbers = [n for n in parsed_numbers if n is not None and n > 0]

                if len(parsed_numbers) >= 2:
                    description = " ".join(current_item_desc_parts).strip()
                    if not description:
                         temp_desc_line = line_text
                         for num_str in numbers_on_line: temp_desc_line = temp_desc_line.replace(num_str, " ")
                         description = re.sub(r'[₹$€£,\s_]{2,}', ' ', temp_desc_line).strip()
                    current_item_desc_parts = [] 
                    qty, rate, amount = 1.0, 0.0, 0.0
                    parsed_numbers.sort(reverse=True)
                    if len(parsed_numbers) >= 3:
                        amount = parsed_numbers[0]
                        rate = parsed_numbers[1]
                        qty = parsed_numbers[2]
                        if abs(rate * qty - amount) > max(amount, rate*qty) * 0.1 and rate >0 and qty > 0 :
                            if qty == int(qty) and qty <=10: 
                                rate = amount / qty if qty > 0 else 0
                    elif len(parsed_numbers) == 2:
                        amount = parsed_numbers[0]
                        if parsed_numbers[1] == int(parsed_numbers[1]) and parsed_numbers[1] > 0 and parsed_numbers[1] <= 100: 
                            qty = parsed_numbers[1]
                            rate = amount / qty if qty > 0 else 0
                        else: 
                            rate = parsed_numbers[1]
                            qty = 1.0 
                    
                    if not description and (amount > 0 or rate > 0): description = "Misc. Item"
                    if not description and (qty == 1.0 and rate == 0.0 and amount == 0.0): continue 

                    mapped_items.append({
                        'description': description if description else "Item (No Description)", 
                        'quantity': qty, 'rate': rate, 'amount': amount,
                        'gst_rate': 0.0, 'gst_amount': 0.0, 'total_amount': amount
                    })
                else: 
                    temp_line_for_check = re.sub(r'[₹$€£,\s\d\.]', '', line_text) 
                    if len(temp_line_for_check) > 2: 
                        current_item_desc_parts.append(line_text)
        if current_item_desc_parts:
            final_desc = " ".join(current_item_desc_parts).strip()
            if final_desc and len(final_desc) > 5 :
                 mapped_items.append({
                        'description': final_desc, 'quantity': 1.0, 'rate': 0.0, 'amount': 0.0,
                        'gst_rate': 0.0, 'gst_amount': 0.0, 'total_amount': 0.0
                    })

        mapped['items'] = mapped_items
        return mapped

    def _match_line_items_sbert(self, items1, items2):
        if not self.sbert_matcher or not items1 or not items2:
            return [], set(range(len(items1))), set(range(len(items2)))
        desc1 = [str(item.get('description', '')).strip() for item in items1]
        desc2 = [str(item.get('description', '')).strip() for item in items2]
        valid_items1_indices = [i for i, d in enumerate(desc1) if d]
        valid_items2_indices = [i for i, d in enumerate(desc2) if d]
        if not valid_items1_indices or not valid_items2_indices:
            return [], set(range(len(items1))), set(range(len(items2)))
        emb1 = self.sbert_matcher.encode([desc1[i] for i in valid_items1_indices], convert_to_tensor=True)
        emb2 = self.sbert_matcher.encode([desc2[i] for i in valid_items2_indices], convert_to_tensor=True)
        cos_scores = sbert_util.cos_sim(emb1, emb2)
        matched_pairs = []
        used_item2_emb_indices = set()
        for i1_emb_idx, original_item1_idx in enumerate(valid_items1_indices):
            best_score = -1
            best_item2_emb_idx = -1
            for i2_emb_idx, _ in enumerate(valid_items2_indices):
                if i2_emb_idx in used_item2_emb_indices:
                    continue
                score = cos_scores[i1_emb_idx][i2_emb_idx].item()
                if score > best_score:
                    best_score = score
                    best_item2_emb_idx = i2_emb_idx
            if best_item2_emb_idx != -1 and best_score >= self.sbert_item_match_threshold:
                original_item2_idx = valid_items2_indices[best_item2_emb_idx]
                matched_pairs.append({
                    'item1_idx': original_item1_idx, 
                    'item2_idx': original_item2_idx,
                    'score': best_score
                })
                used_item2_emb_indices.add(best_item2_emb_idx)
        unmatched_item1_indices = set(range(len(items1))) - {p['item1_idx'] for p in matched_pairs}
        unmatched_item2_indices = set(range(len(items2))) - {p['item2_idx'] for p in matched_pairs}
        return matched_pairs, unmatched_item1_indices, unmatched_item2_indices

    def _compare_basic_fields(self, inv1, inv2, res):
        for f in ['invoice_number','invoice_date','supplier_name','supplier_gstin','buyer_name','buyer_gstin']:
            v1_raw, v2_raw = inv1.get(f), inv2.get(f)
            is_ocr_error1 = isinstance(v1_raw, str) and ("OCR Error:" in v1_raw or "Error during QA" in v1_raw)
            is_ocr_error2 = isinstance(v2_raw, str) and ("OCR Error:" in v2_raw or "Error during QA" in v2_raw)
            if is_ocr_error1 and is_ocr_error2:
                res['mismatches'][f] = {'invoice1': "OCR Error", 'invoice2': "OCR Error", 'status': 'OCR_ERROR_BOTH'}
                continue
            elif is_ocr_error1:
                res['mismatches'][f] = {'invoice1': "OCR Error", 'invoice2': v2_raw or "N/A", 'status': 'MISMATCH_OCR_ERROR_1'}
                continue
            elif is_ocr_error2:
                res['mismatches'][f] = {'invoice1': v1_raw or "N/A", 'invoice2': "OCR Error", 'status': 'MISMATCH_OCR_ERROR_2'}
                continue
            v1_str = str(v1_raw).strip().upper() if v1_raw is not None else ""
            v2_str = str(v2_raw).strip().upper() if v2_raw is not None else ""
            if v1_raw is None and v2_raw is None: continue
            if f == 'supplier_gstin' or f == 'buyer_gstin':
                _, v1_comp = self.validate_gstin(v1_raw)
                _, v2_comp = self.validate_gstin(v2_raw)
                v1_str_comp = v1_comp.upper() if v1_comp else ""
                v2_str_comp = v2_comp.upper() if v2_comp else ""
            else:
                v1_str_comp = v1_str
                v2_str_comp = v2_str
            if v1_str_comp == v2_str_comp and v1_str_comp != "":
                res['matches'][f] = {'value': v1_raw, 'status': 'MATCH'}
            else:
                res['mismatches'][f] = {'invoice1': v1_raw or "N/A", 'invoice2': v2_raw or "N/A", 'status': 'MISMATCH'}

    def _compare_gst_details(self, inv1, inv2, res):
        for f in ['supplier_gstin','buyer_gstin']:
            g1_raw, g2_raw = inv1.get(f), inv2.get(f)
            is_ocr_error1 = isinstance(g1_raw, str) and ("OCR Error:" in g1_raw or "Error during QA" in g1_raw)
            is_ocr_error2 = isinstance(g2_raw, str) and ("OCR Error:" in g2_raw or "Error during QA" in g2_raw)
            g1_to_validate = None if is_ocr_error1 else g1_raw
            g2_to_validate = None if is_ocr_error2 else g2_raw
            valid1, clean_g1 = self.validate_gstin(g1_to_validate)
            valid2, clean_g2 = self.validate_gstin(g2_to_validate)
            display_g1 = "OCR Error" if is_ocr_error1 else (g1_raw or "N/A")
            display_g2 = "OCR Error" if is_ocr_error2 else (g2_raw or "N/A")
            if g1_to_validate or g2_to_validate:
                 res['gst_calculations'][f'{f}_validation']={
                    'invoice1_valid': valid1, 'invoice2_valid': valid2,
                    'gstin1': display_g1, 'gstin2': display_g2,
                    'clean_gstin1': clean_g1, 'clean_gstin2': clean_g2
                }

    def _reconcile_amounts(self, inv1, inv2, res):
        for f in ['subtotal','total_gst', 'total_cgst', 'total_sgst', 'total_igst', 'grand_total']:
            v1_raw, v2_raw = inv1.get(f), inv2.get(f)
            is_ocr_error1 = isinstance(inv1.get(f'{f}_ocr_error'), str)
            is_ocr_error2 = isinstance(inv2.get(f'{f}_ocr_error'), str)
            if is_ocr_error1 and is_ocr_error2:
                res['mismatches'][f] = {'invoice1': "OCR Error", 'invoice2': "OCR Error", 'difference': "N/A", 'status': 'OCR_ERROR_BOTH'}
                continue
            a1 = v1_raw if isinstance(v1_raw, (int,float)) else 0.0
            a2 = v2_raw if isinstance(v2_raw, (int,float)) else 0.0
            if a1 == 0.0 and a2 == 0.0 and f != 'grand_total' and not is_ocr_error1 and not is_ocr_error2:
                 continue
            diff=abs(a1-a2)
            if diff <= self.tolerance * max(abs(a1), abs(a2), 1.0):
                status_val = 'MATCH_OCR_ERROR_1' if is_ocr_error1 else ('MATCH_OCR_ERROR_2' if is_ocr_error2 else 'MATCH')
                res['matches'][f]={'invoice1': "OCR Error" if is_ocr_error1 else a1,
                                   'invoice2': "OCR Error" if is_ocr_error2 else a2,
                                   'difference':diff,'status':status_val}
            else:
                status_val = 'MISMATCH_OCR_ERROR_1' if is_ocr_error1 else ('MISMATCH_OCR_ERROR_2' if is_ocr_error2 else 'MISMATCH')
                res['mismatches'][f]={'invoice1': "OCR Error" if is_ocr_error1 else a1,
                                      'invoice2': "OCR Error" if is_ocr_error2 else a2,
                                      'difference':diff,'status':status_val}

    def _reconcile_line_items(self, items1, items2, res):
        res['item_reconciliation'] = {
            'total_items_inv1': len(items1), 'total_items_inv2': len(items2),
            'item_matches': [], 'unmatched_items': []
        }
        sbert_paired_items, unmatched1_indices, unmatched2_indices = self._match_line_items_sbert(items1, items2)
        for pair_info in sbert_paired_items:
            it1 = items1[pair_info['item1_idx']]
            it2 = items2[pair_info['item2_idx']]
            comparison_details = self._compare_single_item(it1, it2)
            res['item_reconciliation']['item_matches'].append({
                'item1': it1, 'item2': it2,
                'comparison': comparison_details, 'score': pair_info['score']
            })
        for idx1 in unmatched1_indices:
            res['item_reconciliation']['unmatched_items'].append({'source': 'invoice1', 'item': items1[idx1]})
        for idx2 in unmatched2_indices:
            res['item_reconciliation']['unmatched_items'].append({'source': 'invoice2', 'item': items2[idx2]})

    def _compare_single_item(self,it1,it2):
        comp={'matches':{},'mismatches':{}}
        for f in ['quantity','rate','amount', 'total_amount']:
            v1_raw, v2_raw = it1.get(f), it2.get(f)
            if v1_raw is None and v2_raw is None: continue
            v1 = self._parse_float(v1_raw)
            v2 = self._parse_float(v2_raw)
            if (v1_raw is None and v2_raw is None and v1==0 and v2==0) or (v1 == 0 and v2 == 0 and v1_raw is not None and v2_raw is not None):
                 continue
            diff=abs(v1-v2)
            field_tolerance = self.tolerance * max(abs(v1),abs(v2), 1.0) if 'amount' in f or 'rate' in f else \
                              0.5 if f == 'quantity' else 0.01
            if diff <= field_tolerance:
                comp['matches'][f]={'value1':v1,'value2':v2,'difference':diff}
            else:
                comp['mismatches'][f]={'value1':v1,'value2':v2,'difference':diff}
        return comp

    def _get_expected_gst_profile_for_item(self, description):
        global df_merged_gst_finder, sbert_model_gst_finder, corpus_embeddings_gst_finder, qa_pipeline_gst_finder, all_descriptions_for_fuzzy_gst_finder
        if df_merged_gst_finder.empty: return {"error": "GST Finder: DB not loaded."}
        desc_str = str(description).lower().strip()
        if not desc_str: return {"error": "GST Finder: Item description is empty."}
        retrieved_item_info = None
        if sbert_model_gst_finder and corpus_embeddings_gst_finder is not None and corpus_embeddings_gst_finder.nelement() > 0:
            try:
                query_embedding = sbert_model_gst_finder.encode(desc_str, convert_to_tensor=True)
                cos_scores = sbert_util.cos_sim(query_embedding, corpus_embeddings_gst_finder)[0]
                if len(cos_scores) > 0:
                    top_result = torch.topk(cos_scores, k=1)
                    sbert_score, sbert_idx = top_result.values[0].item(), top_result.indices[0].item()
                    sbert_threshold = 0.40
                    if sbert_score >= sbert_threshold:
                        retrieved_item_info = {'row': df_merged_gst_finder.iloc[sbert_idx], 'score': sbert_score, 'match_type': 'SBERT (GST Profile)'}
            except Exception as e: print(f"Error SBERT in _get_expected_gst_profile: {e}")
        if retrieved_item_info is None and all_descriptions_for_fuzzy_gst_finder:
            best_fuzzy = fuzzy_process.extractOne(desc_str, all_descriptions_for_fuzzy_gst_finder)
            if best_fuzzy:
                fuzzy_desc, fuzzy_score_0_100 = best_fuzzy[0], best_fuzzy[1]
                fuzzy_score = fuzzy_score_0_100 / 100.0
                fuzzy_threshold_internal = 0.75
                if fuzzy_score >= fuzzy_threshold_internal:
                    matched_rows = df_merged_gst_finder[df_merged_gst_finder['Combined_Description'] == fuzzy_desc]
                    if not matched_rows.empty:
                        retrieved_item_info = {'row': matched_rows.iloc[0], 'score': fuzzy_score, 'match_type': 'Fuzzy (GST Profile)'}
        if retrieved_item_info is None:
            return {"error": f"Could not confidently match '{desc_str}' in DB for GST profile.", "db_description": desc_str}
        row_from_db = retrieved_item_info['row']
        match_score_val = retrieved_item_info['score']
        match_type_val = retrieved_item_info['match_type']
        db_desc_match = row_from_db.get('Combined_Description', 'N/A')
        retrieved_hs_code_specific = str(row_from_db.get('HS_Code', 'N/A')).strip()
        cgst_db_val = self._parse_float(row_from_db.get('CGST_Rate', 0.0))
        sgst_db_val = self._parse_float(row_from_db.get('SGST_Rate', 0.0))
        igst_db_val = self._parse_float(row_from_db.get('IGST_Rate', 0.0))
        is_exempted_db_val = bool(row_from_db.get('Is_Exempted', False))
        final_cgst, final_sgst, final_igst, final_exempted = cgst_db_val, sgst_db_val, igst_db_val, is_exempted_db_val
        if qa_pipeline_gst_finder and match_score_val > 0.5:
            context = (f"Product: '{db_desc_match}', HS Code: {retrieved_hs_code_specific}. "
                       f"CGST: {cgst_db_val}%. SGST: {sgst_db_val}%. IGST: {igst_db_val}%. "
                       f"Exempted status: {'yes' if is_exempted_db_val else 'no'}.")
            qa_map_gst = {"cgst_qa": "What is CGST rate?", "sgst_qa": "What is SGST rate?",
                          "igst_qa": "What is IGST rate?", "is_exempted_qa": "Is the product exempted from tax?"}
            try:
                for key, q_text in qa_map_gst.items():
                    ans = qa_pipeline_gst_finder(question=q_text, context=context)
                    ans_text = ans['answer']
                    if "rate" in q_text:
                        num_match = re.search(r'(\d+\.?\d*)', ans_text)
                        val = self._parse_float(num_match.group(1)) if num_match else None
                        if val is not None:
                            if key == "cgst_qa": final_cgst = val
                            elif key == "sgst_qa": final_sgst = val
                            elif key == "igst_qa": final_igst = val
                    elif key == "is_exempted_qa":
                        final_exempted = any(kw in ans_text.lower() for kw in ["yes", "exempt", "true"])
            except Exception as e:
                print(f"QA pipeline error during GST profile fetch: {e}")
        return {"db_description": db_desc_match, "hs_code_db": retrieved_hs_code_specific,
                "score_db": match_score_val, "match_type_db": match_type_val,
                "igst_db": final_igst, "cgst_db": final_cgst, "sgst_db": final_sgst,
                "is_exempted_db": final_exempted}

    def verify_item_gst_rates(self, item_inv):
        item_desc = str(item_inv.get('description','')).strip()
        exp_prof = self._get_expected_gst_profile_for_item(item_desc)
        status_dets = {
            "invoice_item_description": item_desc if item_desc else "N/A",
            "db_match_description": exp_prof.get("db_description", "N/A"),
            "db_match_score": f"{exp_prof.get('score_db', 0.0):.2f} ({exp_prof.get('match_type_db','N/A')})" if "score_db" in exp_prof else "N/A",
            "db_hs_code": exp_prof.get("hs_code_db", "N/A"),
            "expected_igst_db": exp_prof.get("igst_db", "N/A"),
            "expected_cgst_db": exp_prof.get("cgst_db", "N/A"),
            "expected_sgst_db": exp_prof.get("sgst_db", "N/A"),
            "is_exempted_db": exp_prof.get("is_exempted_db", "Unknown"),
            "status": "UNVERIFIED_DB_LOOKUP"
        }
        if "error" in exp_prof:
            status_dets["status"] = f"DB_LOOKUP_FAILED ({exp_prof['error']})"
            return status_dets
        db_i = self._parse_float(exp_prof.get("igst_db", 0.0))
        db_c = self._parse_float(exp_prof.get("cgst_db", 0.0))
        db_s = self._parse_float(exp_prof.get("sgst_db", 0.0))
        eff_db_rate = db_i if db_i > 0 else (db_c + db_s)
        if exp_prof.get("is_exempted_db") == True:
            eff_db_rate = 0.0
            status_dets["status"] = "EXPECTED_EXEMPTED_IN_DB"
        elif eff_db_rate > 0 :
            status_dets["status"] = f"EXPECTED_RATE_IN_DB ({eff_db_rate}%)"
        else:
            status_dets["status"] = "DB_RATE_ZERO_OR_UNCLEAR"
        status_dets["effective_expected_rate_db"] = f"{eff_db_rate}%"
        return status_dets

    def compare_invoices(self,d1_ocr,d2_ocr,comp_type="buyer_seller_ocr"):
        res = {'comparison_id':f"CMP_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
               'comparison_type':comp_type,
               'invoice1_orig_data': d1_ocr, 'invoice2_orig_data': d2_ocr,
               'invoice1_mapped': self.map_ocr_output_to_reconciler_format(d1_ocr),
               'invoice2_mapped': self.map_ocr_output_to_reconciler_format(d2_ocr),
               'matches':{}, 'mismatches':{}, 'gst_calculations':{},
               'item_reconciliation':{},
               'item_gst_authenticity_inv1':[], 'item_gst_authenticity_inv2':[],
               'overall_status':'UNKNOWN', 'confidence_score':0.0, 'summary':{}
              }
        if res['invoice1_mapped'].get('ocr_main_error') and res['invoice2_mapped'].get('ocr_main_error'):
            res['overall_status'] = 'OCR_FAILED_BOTH_INVOICES'
            res['confidence_score'] = 0.0
            res['summary'] = {'total_fields_compared': 0, 'fields_matched': 0, 'fields_mismatched': 0,
                              'critical_mismatches': ['OCR Failed for Both']}
            return res
        elif res['invoice1_mapped'].get('ocr_main_error'):
            res['overall_status'] = 'OCR_FAILED_INVOICE_1'
        elif res['invoice2_mapped'].get('ocr_main_error'):
            res['overall_status'] = 'OCR_FAILED_INVOICE_2'
        self._compare_basic_fields(res['invoice1_mapped'],res['invoice2_mapped'],res)
        self._compare_gst_details(res['invoice1_mapped'],res['invoice2_mapped'],res)
        self._reconcile_amounts(res['invoice1_mapped'],res['invoice2_mapped'],res)
        itms1 = res['invoice1_mapped'].get('items',[])
        itms2 = res['invoice2_mapped'].get('items',[])
        itms1 = itms1 if isinstance(itms1, list) else []
        itms2 = itms2 if isinstance(itms2, list) else []
        if itms1 or itms2:
            self._reconcile_line_items(itms1, itms2, res)
        for it1 in itms1: res['item_gst_authenticity_inv1'].append(self.verify_item_gst_rates(it1))
        for it2 in itms2: res['item_gst_authenticity_inv2'].append(self.verify_item_gst_rates(it2))
        self._calculate_overall_status(res)
        self._generate_summary(res)
        return res

    def _calculate_overall_status(self,res):
        if "OCR_FAILED" in res['overall_status'] and not (res['matches'] or res['mismatches']):
            res['confidence_score'] = 0.0
            return
        num_matches = len(res['matches'])
        num_mismatches = len(res['mismatches'])
        comparable_header_fields = 0
        header_field_keys = ['invoice_number','invoice_date','supplier_name','supplier_gstin',
                             'buyer_name','buyer_gstin','subtotal','total_gst','grand_total']
        for key in header_field_keys:
            v1 = res['invoice1_mapped'].get(key)
            v2 = res['invoice2_mapped'].get(key)
            err1 = isinstance(v1, str) and ("OCR Error:" in v1 or "Error during QA" in v1)
            err2 = isinstance(v2, str) and ("OCR Error:" in v2 or "Error during QA" in v2)
            if not (err1 and err2):
                if v1 is not None or v2 is not None :
                    comparable_header_fields += 1
        if comparable_header_fields == 0 and not (res['invoice1_mapped'].get('items') or res['invoice2_mapped'].get('items')):
            res['overall_status']='NO_VALID_DATA_FOR_COMPARISON'
            res['confidence_score']=0.0
            return
        header_match_score = (num_matches / comparable_header_fields) * 100 if comparable_header_fields > 0 else 0.0
        item_rec = res.get('item_reconciliation', {})
        total_items1 = item_rec.get('total_items_inv1', 0)
        total_items2 = item_rec.get('total_items_inv2', 0)
        matched_item_pairs = item_rec.get('item_matches', [])
        num_sbert_matched_pairs = len(matched_item_pairs)
        item_match_quality_score = 0
        if num_sbert_matched_pairs > 0:
            avg_sbert_score = sum(p.get('score', 0) for p in matched_item_pairs) / num_sbert_matched_pairs
            item_match_quality_score = avg_sbert_score * 100
        max_possible_item_matches = max(total_items1, total_items2)
        item_proportion_score = (num_sbert_matched_pairs / max_possible_item_matches) * 100 if max_possible_item_matches > 0 else 0.0
        overall_item_score = (item_proportion_score * 0.5) + (item_match_quality_score * 0.5) if (total_items1 > 0 or total_items2 > 0) else 100.0
        current_confidence = (header_match_score * 0.6) + (overall_item_score * 0.4)
        critical_fields = ['grand_total','supplier_gstin']
        critical_mismatch_penalty = 0
        for f in critical_fields:
            if f in res['mismatches']:
                mismatch_data = res['mismatches'][f]
                v1_crit = mismatch_data.get('invoice1')
                v2_crit = mismatch_data.get('invoice2')
                err_v1_crit = isinstance(v1_crit, str) and ("OCR Error" in v1_crit or "N/A" == v1_crit)
                err_v2_crit = isinstance(v2_crit, str) and ("OCR Error" in v2_crit or "N/A" == v2_crit)
                if not (err_v1_crit and err_v2_crit):
                    critical_mismatch_penalty += 25
        current_confidence -= critical_mismatch_penalty
        res['confidence_score'] = round(max(0, min(current_confidence, 100)), 1)
        has_critical_mismatches = critical_mismatch_penalty > 0
        if res['confidence_score'] >= 90 and not has_critical_mismatches and num_mismatches == 0 :
            res['overall_status'] = 'PERFECT_MATCH'
        elif res['confidence_score'] >= 75 and not has_critical_mismatches:
            res['overall_status'] = 'GOOD_MATCH'
        elif res['confidence_score'] >= 50:
            res['overall_status'] = 'PARTIAL_MATCH'
            if has_critical_mismatches: res['overall_status'] += ' (Critical Mismatch)'
        else:
            res['overall_status'] = 'POOR_MATCH'
            if has_critical_mismatches: res['overall_status'] += ' (Critical Mismatch)'
        if "OCR_FAILED" in res['overall_status']:
            pass
        elif res['invoice1_mapped'].get('ocr_main_error') or res['invoice2_mapped'].get('ocr_main_error'):
            res['overall_status'] = 'POOR_MATCH (OCR Issues)'

    def _generate_summary(self,res):
        valid_mismatches_for_count = {
            k: v for k, v in res['mismatches'].items()
            if not (isinstance(v.get('invoice1'), str) and "OCR Error" in v.get('invoice1') and
                    isinstance(v.get('invoice2'), str) and "OCR Error" in v.get('invoice2'))
        }
        res['summary']={
            'total_fields_compared': len(res['matches']) + len(valid_mismatches_for_count),
            'fields_matched': len(res['matches']),
            'fields_mismatched': len(valid_mismatches_for_count),
            'amount_differences':[],
            'critical_mismatches':[]
        }
        for f,d_val in valid_mismatches_for_count.items():
            if 'difference' in d_val and isinstance(d_val['difference'],(int,float)):
                res['summary']['amount_differences'].append({'field':f,'difference':d_val['difference']})
        critical_fields_check = ['grand_total','supplier_gstin']
        for f_crit in critical_fields_check:
            if f_crit in valid_mismatches_for_count:
                 res['summary']['critical_mismatches'].append(f_crit.replace("_"," ").title())

    def get_html_template(self):
        # Dark Theme CSS
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GST Reconciliation Report</title>
            <style>
                :root {
                    --bg-color: #1a202c; /* Very Dark Blue/Grey */
                    --card-bg-color: #2d3748; /* Dark Blue/Grey */
                    --text-color: #e2e8f0; /* Light Grey/Off-white */
                    --text-muted-color: #a0aec0; /* Medium Grey */
                    --border-color: #4a5568; /* Grey */
                    --primary-accent: #4299e1; /* Blue */
                    --match-color: #68d391; /* Green */
                    --mismatch-color: #fc8181; /* Red */
                    --ocr-error-color: #f6ad55; /* Orange */
                    --neutral-color: var(--text-muted-color);
                }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: var(--bg-color);
                    color: var(--text-color);
                    line-height: 1.6;
                }
                .report-container {
                    max-width: 1200px;
                    margin: 20px auto;
                    background: var(--card-bg-color);
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.25); /* Darker shadow for dark theme */
                    border: 1px solid var(--border-color);
                }
                h1, h2, h3 {
                    color: var(--text-color);
                    margin-top: 0;
                }
                h1 {
                    text-align: center;
                    font-size: 2em;
                    margin-bottom: 10px;
                    color: var(--primary-accent); 
                }
                .report-meta {
                    text-align: center;
                    font-size: 0.9em;
                    color: var(--text-muted-color);
                    margin-bottom: 30px;
                }
                .status-badge-container { text-align: center; margin-bottom: 25px; }
                .status-badge {
                    display: inline-block;
                    padding: 10px 20px;
                    border-radius: 20px;
                    color: #1a202c; /* Dark text on light badges for contrast */
                    font-weight: 600;
                    font-size: 1.1em;
                    letter-spacing: 0.5px;
                }
                /* Status Badge Backgrounds - Light colors for dark theme */
                .status-perfect_match { background-color: var(--match-color); color: #1A202C; }
                .status-good_match { background-color: #9ae6b4; color: #1A202C; } /* Lighter Green */
                .status-partial_match { background-color: #fbd38d; color: #1A202C; } /* Light Orange */
                .status-poor_match { background-color: var(--mismatch-color); color: #1A202C; }
                .status-partial_match-critical-mismatch,
                .status-poor_match-critical-mismatch { background-color: #e53e3e; color: #fff; } /* Darker Red with white text */
                .status-no_data_for_comparison, .status-ocr_failed_both_invoices,
                .status-ocr_failed_invoice_1, .status-ocr_failed_invoice_2,
                .status-no_valid_data_for_comparison, .status-poor_match-ocr-issues { background-color: #718096; color: #fff; }

                .section { margin-bottom: 30px; padding: 20px; background-color: #222b39; border-radius:8px; border: 1px solid var(--border-color); }
                .section h2 { font-size: 1.5em; margin-bottom: 15px; border-bottom: 2px solid var(--border-color); padding-bottom:10px; color: var(--primary-accent);}
                
                .grid-2-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom:20px;}
                .card { background: var(--card-bg-color); padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid var(--border-color);}
                .card h3 { font-size: 1.2em; margin-bottom: 10px; color: var(--primary-accent); }
                .card p { margin: 5px 0; font-size: 0.95em; }
                .card strong { color: #cbd5e0; } /* Lighter grey for strong text */

                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                    font-size: 0.9em;
                }
                th, td {
                    border: 1px solid var(--border-color);
                    padding: 12px 15px;
                    text-align: left;
                    vertical-align: top;
                }
                th {
                    background-color: #2a3343; /* Darker header for dark theme */
                    color: #cbd5e0; /* Lighter text for header */
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                tr:nth-child(even) td { background-color: #27303f; }
                tr:hover td { background-color: #323c4e; }

                .match { color: var(--match-color); font-weight: bold; }
                .mismatch { color: var(--mismatch-color); font-weight: bold; }
                .neutral, .status-not_applicable { color: var(--text-muted-color); }
                .ocr-error-text { color: var(--ocr-error-color); font-style: italic; font-size: 0.9em; }
                .critical-mismatch-field { font-weight: bold; color: var(--mismatch-color); }

                .summary-item {
                    background-color: #2a3343; border-left: 5px solid var(--primary-accent);
                    padding: 12px 18px; margin-bottom: 10px; border-radius: 4px;
                    font-size: 0.95em;
                }
                .summary-item.critical { border-left-color: var(--mismatch-color); background-color: #4c3030; } /* Dark red background */
                .summary-item.good { border-left-color: var(--match-color); background-color: #2f4c39; } /* Dark green background */
                
                /* GST Authenticity Status specific styles */
                .status-expected_exempted_in_db { color: var(--match-color); font-weight: 500;}
                .status-expected_rate_in_db { color: #63b3ed; font-weight: 500;} /* Light Blue */
                .status-db_rate_zero_or_unclear { color: var(--text-muted-color); }
                .status-db_lookup_failed { color: var(--ocr-error-color); font-weight: bold; }
                .status-unverified_db_lookup { color: var(--text-muted-color); }

                @media (max-width: 768px) {
                    .grid-2-col { grid-template-columns: 1fr; }
                    h1 { font-size: 1.8em; }
                    .section h2 { font-size: 1.3em; }
                    th, td { padding: 8px 10px; }
                }
            </style>
        </head>
        <body>
            <div class="report-container">
                <h1>GST Reconciliation Report</h1>
                <p class="report-meta">Comparison ID: <strong>{{COMPARISON_ID}}</strong> | Generated: {{GENERATED_DATE}}</p>
                
                <div class="status-badge-container">
                    <span class="status-badge status-{{OVERALL_STATUS_CLASS}}">
                        Overall Status: {{OVERALL_STATUS}} (Confidence: {{CONFIDENCE_SCORE}}%)
                    </span>
                </div>

                <div class="section">
                    <h2>Overall Summary</h2>
                    <div class="summary-item">
                        Fields Compared (Header/Totals): <strong>{{TOTAL_FIELDS}}</strong> | 
                        Matched: <span class="match">{{FIELDS_MATCHED}}</span> | 
                        Mismatched: <span class="mismatch">{{FIELDS_MISMATCHED}}</span>
                    </div>
                    <div class="summary-item">Comparison Type: <strong>{{COMPARISON_TYPE}}</strong></div>
                    {% if CRITICAL_MISMATCHES_STR %}
                        <div class="summary-item critical">
                            Critical Mismatches found in: <span class="critical-mismatch-field">{{CRITICAL_MISMATCHES_STR}}</span>
                        </div>
                    {% else %}
                        <div class="summary-item good">No critical header/total mismatches detected.</div>
                    {% endif %}
                </div>

                <div class="section">
                    <h2>Invoice Source Details</h2>
                    <div class="grid-2-col">
                        <div class="card">
                            <h3>Invoice 1: {{INV1_FILENAME}}</h3>
                            <p><strong>Inv No:</strong> {{INV1_NUMBER}}</p>
                            <p><strong>Date:</strong> {{INV1_DATE}}</p>
                            <p><strong>Supplier:</strong> {{INV1_SUPPLIER_NAME}}</p>
                            <p><strong>Supplier GSTIN:</strong> {{INV1_SUPPLIER_GSTIN}}</p>
                            <p><strong>Buyer:</strong> {{INV1_BUYER_NAME}}</p>
                            <p><strong>Buyer GSTIN:</strong> {{INV1_BUYER_GSTIN}}</p>
                            <p><strong>Grand Total:</strong> {{INV1_GRAND_TOTAL}}</p>
                        </div>
                        <div class="card">
                            <h3>Invoice 2: {{INV2_FILENAME}}</h3>
                            <p><strong>Inv No:</strong> {{INV2_NUMBER}}</p>
                            <p><strong>Date:</strong> {{INV2_DATE}}</p>
                            <p><strong>Supplier:</strong> {{INV2_SUPPLIER_NAME}}</p>
                            <p><strong>Supplier GSTIN:</strong> {{INV2_SUPPLIER_GSTIN}}</p>
                            <p><strong>Buyer:</strong> {{INV2_BUYER_NAME}}</p>
                            <p><strong>Buyer GSTIN:</strong> {{INV2_BUYER_GSTIN}}</p>
                            <p><strong>Grand Total:</strong> {{INV2_GRAND_TOTAL}}</p>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>Field-by-Field Comparison (Header/Totals)</h2>
                    <table>
                        <thead><tr><th>Field</th><th>Invoice 1 Value</th><th>Invoice 2 Value</th><th>Status</th><th>Difference</th></tr></thead>
                        <tbody>{{COMPARISON_ROWS}}</tbody>
                    </table>
                </div>
                
                {{ITEM_RECONCILIATION_SECTION}}
                {{ITEM_GST_AUTHENTICITY_SECTION_INV1}}
                {{ITEM_GST_AUTHENTICITY_SECTION_INV2}}

                <div class="section">
                    <h2>GSTIN Validation</h2>
                    <table>
                        <thead><tr><th>GSTIN Type</th><th>Inv1 GSTIN (Validation)</th><th>Inv2 GSTIN (Validation)</th><th>Status</th></tr></thead>
                        <tbody>{{GSTIN_VALIDATION_ROWS}}</tbody>
                    </table>
                </div>

            </div>
        </body>
        </html>
        """

    def _format_report_value(self, value, is_amount=False, is_gstin_validation_text=False):
        if isinstance(value, str) and ("OCR Error:" in value or "Error during QA" in value):
            return f'<span class="ocr-error-text">{value}</span>'
        if value is None or value == "N/A" or (isinstance(value, float) and np.isnan(value)):
             return "N/A" if not is_gstin_validation_text else "N/A (Not Provided)"
        if is_amount:
            try:
                num_val = self._parse_float(value)
                return f"₹{num_val:,.2f}"
            except:
                return str(value)
        return str(value)

    def generate_reconciliation_report(self, comparison_result, inv1_filename="Invoice1", inv2_filename="Invoice2"):
        html_template = self.get_html_template()
        status_map = {'PERFECT_MATCH':'perfect_match', 'GOOD_MATCH':'good_match',
                      'PARTIAL_MATCH':'partial_match', 'POOR_MATCH':'poor_match',
                      'PARTIAL_MATCH (Critical Mismatch)':'partial_match-critical-mismatch',
                      'POOR_MATCH (Critical Mismatch)':'poor_match-critical-mismatch',
                      'NO_DATA_FOR_COMPARISON':'no_data_for_comparison',
                      'OCR_FAILED_BOTH_INVOICES': 'ocr_failed_both_invoices',
                      'OCR_FAILED_INVOICE_1': 'ocr_failed_invoice_1',
                      'OCR_FAILED_INVOICE_2': 'ocr_failed_invoice_2',
                      'NO_VALID_DATA_FOR_COMPARISON': 'no_valid_data_for_comparison',
                      'POOR_MATCH (OCR Issues)': 'poor_match-ocr-issues'
                      }
        overall_status_class = status_map.get(comparison_result.get('overall_status', 'poor_match'), 'poor_match')
        inv1_disp = comparison_result['invoice1_mapped']
        inv2_disp = comparison_result['invoice2_mapped']
        comp_rows_html = ""
        header_fields = ['invoice_number','invoice_date','supplier_name','supplier_gstin',
                         'buyer_name','buyer_gstin','subtotal','total_cgst', 'total_sgst',
                         'total_igst', 'total_gst','grand_total']
        for field in header_fields:
            is_amt = any(k in field for k in ['total', 'subtotal', 'amount'])
            v1_report_val = inv1_disp.get(field)
            v2_report_val = inv2_disp.get(field)
            d_v1 = self._format_report_value(v1_report_val, is_amt)
            d_v2 = self._format_report_value(v2_report_val, is_amt)
            status_html = '<span class="neutral">-</span>'
            diff_html = "-"
            if field in comparison_result['matches']:
                match_data = comparison_result['matches'][field]
                matched_val_display = self._format_report_value(match_data.get('value', v1_report_val), is_amt)
                d_v1, d_v2 = matched_val_display, matched_val_display
                status_html = f'<span class="match">{match_data.get("status", "MATCH").replace("_", " ")}</span>'
                if 'difference' in match_data and is_amt:
                     diff_html = f"₹{self._parse_float(match_data['difference']):.2f}"
            elif field in comparison_result['mismatches']:
                mismatch_data = comparison_result['mismatches'][field]
                status_html = f'<span class="mismatch">{mismatch_data.get("status", "MISMATCH").replace("_", " ")}</span>'
                if 'difference' in mismatch_data and is_amt:
                    diff_html = f"₹{self._parse_float(mismatch_data['difference']):.2f}"
            elif inv1_disp.get('ocr_main_error') or inv2_disp.get('ocr_main_error'):
                status_html = '<span class="ocr-error-text">OCR Issue</span>'
            elif d_v1 == "N/A" and d_v2 == "N/A":
                status_html = '<span class="neutral">Both N/A</span>'
            comp_rows_html += f'<tr><td>{field.replace("_"," ").title()}</td><td>{d_v1}</td><td>{d_v2}</td><td>{status_html}</td><td>{diff_html}</td></tr>'
        gst_val_rows_html = ""
        for field_key, data in comparison_result.get('gst_calculations', {}).items():
            if 'validation' in field_key:
                type_name = field_key.replace('_validation','').replace('_',' ').title()
                gstin1_disp = self._format_report_value(data.get('gstin1'), is_gstin_validation_text=True)
                gstin2_disp = self._format_report_value(data.get('gstin2'), is_gstin_validation_text=True)
                valid1_str = "✓ Valid" if data.get('invoice1_valid') else "✗ Invalid"
                valid2_str = "✓ Valid" if data.get('invoice2_valid') else "✗ Invalid"
                status_str, status_cls = "N/A", "neutral"
                clean_g1 = data.get('clean_gstin1')
                clean_g2 = data.get('clean_gstin2')
                if (data.get('gstin1') != "N/A" and data.get('gstin1') is not None and not ("OCR Error" in str(data.get('gstin1')))) and \
                   (data.get('gstin2') != "N/A" and data.get('gstin2') is not None and not ("OCR Error" in str(data.get('gstin2')))):
                    if data['invoice1_valid'] and data['invoice2_valid'] and clean_g1 == clean_g2:
                        status_str, status_cls = "MATCH & VALID", "match"
                    elif clean_g1 == clean_g2:
                        status_str, status_cls = "MATCH (Validation Issue)", "mismatch"
                    else:
                        status_str, status_cls = "MISMATCH", "mismatch"
                elif (data.get('gstin1') != "N/A" and data.get('gstin1') is not None and not ("OCR Error" in str(data.get('gstin1')))) or \
                     (data.get('gstin2') != "N/A" and data.get('gstin2') is not None and not ("OCR Error" in str(data.get('gstin2')))):
                    status_str, status_cls = "MISMATCH (One Missing/Error)", "mismatch"
                elif ("OCR Error" in str(data.get('gstin1'))) or ("OCR Error" in str(data.get('gstin2'))):
                    status_str, status_cls = "OCR Error in one/both", "ocr-error-text"
                gst_val_rows_html += f'<tr><td>{type_name}</td><td>{gstin1_disp} ({valid1_str})</td><td>{gstin2_disp} ({valid2_str})</td><td><span class="{status_cls}">{status_str}</span></td></tr>'
        item_rec_section_html = ""
        item_rec_data = comparison_result.get('item_reconciliation', {})
        if item_rec_data and (item_rec_data.get('item_matches') or item_rec_data.get('unmatched_items')):
            rows = ""
            for match_info in item_rec_data.get('item_matches', []):
                i1, i2, comp_details, score = match_info['item1'], match_info['item2'], match_info['comparison'], match_info['score']
                score_disp = f"{score:.2f}" if score is not None else "N/A"
                rows += f"<tr><td><b>MATCHED (SBERT Sim: {score_disp})</b><br/>"
                rows += f"<i>Inv1:</i> {self._format_report_value(i1.get('description'))}<br/>"
                rows += f"<i>Inv2:</i> {self._format_report_value(i2.get('description'))}</td>"
                rows += f"<td>Qty: {self._format_report_value(i1.get('quantity'))} vs {self._format_report_value(i2.get('quantity'))}<br/>"
                rows += f"Rate: {self._format_report_value(i1.get('rate'),True)} vs {self._format_report_value(i2.get('rate'),True)}<br/>"
                rows += f"Amount: {self._format_report_value(i1.get('amount'),True)} vs {self._format_report_value(i2.get('amount'),True)}</td>"
                comp_html = "".join([f'<span class="match">{fname.replace("_"," ").title()}: Matched</span><br/>' for fname in comp_details.get('matches',{})])
                comp_html += "".join([f'<span class="mismatch">{fname.replace("_"," ").title()}: Mismatch ({self._format_report_value(fdat.get("value1"),True)} vs {self._format_report_value(fdat.get("value2"),True)})</span><br/>' for fname,fdat in comp_details.get('mismatches',{}).items()])
                rows += f"<td>{comp_html if comp_html else 'All fields matched'}</td></tr>"
            for unmatch_info in item_rec_data.get('unmatched_items', []):
                item = unmatch_info['item']
                rows += f"<tr><td><b>UNMATCHED ({unmatch_info['source']})</b><br/>{self._format_report_value(item.get('description'))}</td>"
                rows += f"<td>Qty: {self._format_report_value(item.get('quantity'))}<br/>Rate: {self._format_report_value(item.get('rate'),True)}<br/>Amount: {self._format_report_value(item.get('amount'),True)}</td><td>-</td></tr>"
            item_rec_section_html = f"""<div class="section"><h2>Item-Level Reconciliation</h2>
                                     <p>Items Inv1: {item_rec_data.get('total_items_inv1',0)} | Items Inv2: {item_rec_data.get('total_items_inv2',0)} | Matched Pairs: {len(item_rec_data.get('item_matches',[]))}</p>
                                     <table><thead><tr><th>Item Match Status & Description</th><th>Key Figures (Inv1 vs Inv2)</th><th>Field Comparison Details</th></tr></thead>
                                     <tbody>{rows}</tbody></table></div>"""
        def generate_auth_section_html(auth_results_list, inv_num_str):
            if not auth_results_list: return ""
            auth_rows_html = ""
            for r_data in auth_results_list:
                status_class_raw = str(r_data.get("status","unverified")).lower()
                status_class = re.sub(r'[^a-z0-9-]+', '-', status_class_raw).strip('-')
                auth_rows_html += f"""<tr>
                    <td>{self._format_report_value(r_data.get('invoice_item_description'))}</td>
                    <td>{self._format_report_value(r_data.get('db_match_description'))} (Score: {r_data.get('db_match_score','N/A')}, HS: {r_data.get('db_hs_code','N/A')})</td>
                    <td>IGST: {self._format_report_value(r_data.get('expected_igst_db'))}%<br/>
                        CGST: {self._format_report_value(r_data.get('expected_cgst_db'))}%<br/>
                        SGST: {self._format_report_value(r_data.get('expected_sgst_db'))}%</td>
                    <td>{self._format_report_value(r_data.get('effective_expected_rate_db'))}</td>
                    <td>{self._format_report_value(r_data.get('is_exempted_db'))}</td>
                    <td class='status-{status_class}'>{self._format_report_value(r_data.get('status'))}</td></tr>"""
            return f"""<div class="section"><h2>Item-wise GST Rate Authenticity Check ({inv_num_str} vs DB)</h2>
                       <table><thead><tr><th>Item Desc. ({inv_num_str})</th><th>DB Match (Score, HS)</th>
                       <th>Expected Rates (DB)</th><th>Effective DB Rate</th><th>DB Exempted?</th>
                       <th>Authenticity Status</th></tr></thead><tbody>{auth_rows_html}</tbody></table></div>"""
        item_gst_auth_inv1_html = generate_auth_section_html(comparison_result.get('item_gst_authenticity_inv1', []), "Inv1")
        item_gst_auth_inv2_html = generate_auth_section_html(comparison_result.get('item_gst_authenticity_inv2', []), "Inv2")
        crit_mismatches_str = ", ".join(comparison_result['summary'].get('critical_mismatches', []))
        replacements = {
            '{{COMPARISON_ID}}': comparison_result['comparison_id'],
            '{{GENERATED_DATE}}': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '{{OVERALL_STATUS_CLASS}}': overall_status_class,
            '{{OVERALL_STATUS}}': comparison_result.get('overall_status', 'Error').replace('_',' ').title(),
            '{{CONFIDENCE_SCORE}}': f"{comparison_result.get('confidence_score', 0.0):.1f}",
            '{{TOTAL_FIELDS}}': str(comparison_result['summary'].get('total_fields_compared',0)),
            '{{FIELDS_MATCHED}}': str(comparison_result['summary'].get('fields_matched',0)),
            '{{FIELDS_MISMATCHED}}': str(comparison_result['summary'].get('fields_mismatched',0)),
            '{{COMPARISON_TYPE}}': comparison_result.get('comparison_type','N/A').replace('_',' ').title(),
            '{{CRITICAL_MISMATCHES_STR}}': crit_mismatches_str,
            '{{INV1_FILENAME}}': inv1_filename,
            '{{INV2_FILENAME}}': inv2_filename,
            '{{INV1_NUMBER}}': self._format_report_value(inv1_disp.get('invoice_number')),
            '{{INV1_DATE}}': self._format_report_value(inv1_disp.get('invoice_date')),
            '{{INV1_SUPPLIER_NAME}}': self._format_report_value(inv1_disp.get('supplier_name')),
            '{{INV1_SUPPLIER_GSTIN}}': self._format_report_value(inv1_disp.get('supplier_gstin')),
            '{{INV1_BUYER_NAME}}': self._format_report_value(inv1_disp.get('buyer_name')),
            '{{INV1_BUYER_GSTIN}}': self._format_report_value(inv1_disp.get('buyer_gstin')),
            '{{INV1_GRAND_TOTAL}}': self._format_report_value(inv1_disp.get('grand_total'), True),
            '{{INV2_NUMBER}}': self._format_report_value(inv2_disp.get('invoice_number')),
            '{{INV2_DATE}}': self._format_report_value(inv2_disp.get('invoice_date')),
            '{{INV2_SUPPLIER_NAME}}': self._format_report_value(inv2_disp.get('supplier_name')),
            '{{INV2_SUPPLIER_GSTIN}}': self._format_report_value(inv2_disp.get('supplier_gstin')),
            '{{INV2_BUYER_NAME}}': self._format_report_value(inv2_disp.get('buyer_name')),
            '{{INV2_BUYER_GSTIN}}': self._format_report_value(inv2_disp.get('buyer_gstin')),
            '{{INV2_GRAND_TOTAL}}': self._format_report_value(inv2_disp.get('grand_total'), True),
            '{{COMPARISON_ROWS}}': comp_rows_html,
            '{{GSTIN_VALIDATION_ROWS}}': gst_val_rows_html,
            '{{ITEM_RECONCILIATION_SECTION}}': item_rec_section_html,
            '{{ITEM_GST_AUTHENTICITY_SECTION_INV1}}': item_gst_auth_inv1_html,
            '{{ITEM_GST_AUTHENTICITY_SECTION_INV2}}': item_gst_auth_inv2_html,
        }
        final_html = html_template
        for placeholder, value in replacements.items():
            final_html = final_html.replace(str(placeholder), str(value))
        return final_html

def extract_and_clean_hsn_from_cell(hsn_cell_value):
    if pd.isna(hsn_cell_value): return ''
    s = str(hsn_cell_value).strip();
    if not s: return ''
    match_keyword = re.search(r'(?:HSN|HS CODE|TARIFF)\s*[:\-]?\s*([\d\.]+)', s, re.IGNORECASE)
    if match_keyword:
        code_candidate = match_keyword.group(1); digits_only = re.sub(r'\D', '', code_candidate)
        if 2 <= len(digits_only) <= 8: return digits_only.lower()
    potential_numeric_codes = re.findall(r'[\d\.]+', s)
    for pn_code in potential_numeric_codes:
        digits_only = re.sub(r'\D', '', pn_code)
        if 2 <= len(digits_only) <= 8: return digits_only.lower()
    segments = re.split(r'[,;\s/&]+', s)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        digits_only = re.sub(r'\D', '', segment)
        if 2 <= len(digits_only) <= 8:
            return digits_only.lower()
    all_digits_from_start_match = re.match(r'(\d+)', s)
    if all_digits_from_start_match:
        digits_sequence = all_digits_from_start_match.group(1)
        if len(digits_sequence) >= 4: return digits_sequence[:4].lower()
        elif 2 <= len(digits_sequence) < 4: return digits_sequence.lower()
    return ''

def parse_hsn_pdf_gst_finder(pdf_content_bytes_unused):
    global HSN_PDF_PATH; data = []
    try:
        if not os.path.exists(HSN_PDF_PATH): print(f"GST Finder Error: HSN PDF file not found at '{HSN_PDF_PATH}'"); return pd.DataFrame(columns=['HS_Code_PDF', 'Description_PDF'])
        with open(HSN_PDF_PATH, "rb") as f_in: pdf_content_bytes = f_in.read()
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        header_row_index = -1
                        for r_idx, r_content in enumerate(table): 
                            if r_content and len(r_content) > 2: 
                                col1_text = str(r_content[1] if r_content[1] else "").upper() 
                                col2_text = str(r_content[2] if r_content[2] else "").upper() 
                                if any(kw in col1_text for kw in ['HS CODE', 'HSN']) and 'DESCRIPTION' in col2_text:
                                    header_row_index = r_idx; break
                        data_rows_to_parse = table[header_row_index+1:] if header_row_index != -1 else table
                        for row_idx, row in enumerate(data_rows_to_parse):
                            if len(row) >= 3 :
                                hs_code_raw = str(row[1]).replace('\n',' ').strip() if row[1] else None
                                description_raw = str(row[2]).replace('\n',' ').strip() if row[2] else None
                                if hs_code_raw and description_raw:
                                    cleaned_hs = re.sub(r'[^0-9]', '', hs_code_raw) 
                                    if 2 <= len(cleaned_hs) <= 8:
                                        data.append({'HS_Code_PDF': cleaned_hs, 'Description_PDF': description_raw})
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        match = re.search(r'^\s*(\d[\d\.\s]*\d)\s+([A-Za-z].*)', line) 
                        if match:
                            hs_code_raw = match.group(1).strip(); description_raw = match.group(2).strip()
                            cleaned_hs = re.sub(r'[^0-9]','', hs_code_raw)
                            if cleaned_hs and description_raw and len(description_raw) > 3 and (2 <= len(cleaned_hs) <= 8) :
                                data.append({'HS_Code_PDF': cleaned_hs, 'Description_PDF': description_raw})
        if not data: print("GST Finder Warning: No data extracted from HSN PDF."); return pd.DataFrame(columns=['HS_Code_PDF','Description_PDF'])
        df_hsn = pd.DataFrame(data)
        if df_hsn.empty: return df_hsn
        df_hsn['HS_Code_PDF'] = df_hsn['HS_Code_PDF'].astype(str).str.lower()
        df_hsn.dropna(subset=['HS_Code_PDF', 'Description_PDF'], inplace=True)
        df_hsn = df_hsn[df_hsn['HS_Code_PDF'] != '']
        df_hsn.drop_duplicates(subset=['HS_Code_PDF'], keep='first', inplace=True)
        print(f"GST Finder: Extracted {len(df_hsn)} unique HSN entries from PDF.")
        return df_hsn
    except Exception as e: print(f"GST Finder Error parsing HSN PDF: {e}"); import traceback; traceback.print_exc(); return pd.DataFrame(columns=['HS_Code_PDF','Description_PDF'])

def aggregate_hsn_descriptions_gst_finder(df_hsn_input):
    if df_hsn_input.empty or not all(c in df_hsn_input.columns for c in ['HS_Code_PDF', 'Description_PDF']): print("GST Finder Warning: HSN input DataFrame for aggregation is empty/missing cols."); return df_hsn_input.copy()
    print("GST Finder: Aggregating hierarchical HSN descriptions..."); df_hsn = df_hsn_input.copy()
    df_hsn['HS_Code_PDF'] = df_hsn['HS_Code_PDF'].astype(str); df_hsn.sort_values(by='HS_Code_PDF', inplace=True); df_hsn.reset_index(drop=True, inplace=True)
    aggregated_descriptions = {}; unique_hs_codes_sorted = sorted(df_hsn['HS_Code_PDF'].unique())
    for parent_hs in unique_hs_codes_sorted:
        parent_desc_series = df_hsn[df_hsn['HS_Code_PDF'] == parent_hs]['Description_PDF']
        if parent_desc_series.empty or pd.isna(parent_desc_series.iloc[0]): continue
        current_desc_list = [parent_desc_series.iloc[0]]
        for child_hs in unique_hs_codes_sorted:
            if child_hs.startswith(parent_hs) and len(child_hs) > len(parent_hs) and len(parent_hs) >=2:
                child_desc_series = df_hsn[df_hsn['HS_Code_PDF'] == child_hs]['Description_PDF']
                if not child_desc_series.empty and pd.notna(child_desc_series.iloc[0]): current_desc_list.append(child_desc_series.iloc[0])
        aggregated_descriptions[parent_hs] = ". ".join(list(dict.fromkeys(current_desc_list)))
    df_hsn['Aggregated_Description_PDF'] = df_hsn['HS_Code_PDF'].map(aggregated_descriptions)
    df_hsn['Aggregated_Description_PDF'] = df_hsn['Aggregated_Description_PDF'].fillna(df_hsn['Description_PDF'])
    print("GST Finder: HSN description aggregation complete."); return df_hsn

def parse_gst_csv_gst_finder(csv_content_bytes_unused):
    global GST_CSV_PATH; df_gst_raw = None; encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1', 'windows-1252']
    try:
        if not os.path.exists(GST_CSV_PATH): print(f"--- GST_CSV ERROR: File NOT FOUND at '{GST_CSV_PATH}' ---"); return pd.DataFrame()
        with open(GST_CSV_PATH, "rb") as f_in: csv_bytes_content = f_in.read()
        successful_encoding = None
        for enc in encodings_to_try:
            try:
                df_temp = pd.read_csv(io.BytesIO(csv_bytes_content), encoding=enc, low_memory=False)
                if not df_temp.empty and any(any(kw in str(c).lower() for kw in ['gst','hsn','code','desc','tariff','rate']) for c in df_temp.columns):
                    df_gst_raw, successful_encoding = df_temp, enc
                    print(f"--- GST_CSV INFO: Successfully loaded with encoding: '{successful_encoding}'. Shape: {df_gst_raw.shape}, Cols: {df_gst_raw.columns.tolist()} ---"); break
            except UnicodeDecodeError: print(f"--- GST_CSV WARNING: UnicodeDecodeError with encoding '{enc}'. ---")
            except Exception as e: print(f"--- GST_CSV WARNING: Error with '{enc}': {e}. ---")
        if df_gst_raw is None or df_gst_raw.empty:
            try:
                df_temp = pd.read_csv(io.BytesIO(csv_bytes_content), encoding='utf-8', errors='replace', low_memory=False)
                if not df_temp.empty and any(any(kw in str(c).lower() for kw in ['gst','hsn','code','desc','rate']) for c in df_temp.columns):
                    df_gst_raw = df_temp; print(f"--- GST_CSV WARNING: Loaded with 'utf-8' errors='replace'. Shape: {df_gst_raw.shape} ---")
                else: print("--- GST_CSV CRITICAL: Last resort read also failed. ---"); return pd.DataFrame()
            except Exception as e: print(f"--- GST_CSV CRITICAL: Exception during last resort: {e} ---"); return pd.DataFrame()

        original_hsn_col_name = None; df_cols_map = {str(c).lower().strip().replace('\n', ' '): c for c in df_gst_raw.columns}
        hsn_kws = ['chapter/ heading/ sub-heading/ tariff item', 'chapter/heading/sub-heading/tariffitem', 'hsn', 'tariff code', 'hs code', 'tariffitem', 'tariff item']
        for kw in hsn_kws:
            if kw in df_cols_map: original_hsn_col_name = df_cols_map[kw]; print(f"--- GST_CSV DEBUG: Identified HSN col: '{original_hsn_col_name}' (via '{kw}') ---"); break
        
        if not original_hsn_col_name: print("--- GST_CSV CRIT ERROR: Could not identify original HSN col. ---"); df_gst_raw['HS_Code_GST_Cleaned'] = ''
        else: df_gst_raw['HS_Code_GST_Cleaned'] = df_gst_raw[original_hsn_col_name].apply(extract_and_clean_hsn_from_cell)

        col_map_kws={'Description_GST':['description of goods', 'descriptionofgoods','desc of goods','description','goods desc'],'CGST_Rate':['cgst rate (%)','cgst(%)','cgst rate','cgst'],'SGST_Rate':['sgst/utgst rate (%)','sgst/utgst(%)','sgst rate','sgst','utgst'],'IGST_Rate':['igst rate (%)','igst(%)','igst rate','igst'],'Compensation_Cess_Raw':['compensation cess','compensationcess','cess','comp cess']}
        rename_map = {}
        for target, sources in col_map_kws.items():
            for src_kw in sources:
                if src_kw in df_cols_map:
                    orig_col = df_cols_map[src_kw]
                    if orig_col not in rename_map.values() and orig_col != original_hsn_col_name:
                        rename_map[orig_col] = target; print(f"--- GST_CSV DEBUG: Mapping '{orig_col}' to '{target}' (kw: '{src_kw}') ---"); break
        if rename_map: df_gst_raw.rename(columns=rename_map, inplace=True)
        
        if 'HS_Code_GST_Cleaned' in df_gst_raw.columns: df_gst_raw.rename(columns={'HS_Code_GST_Cleaned': 'HS_Code_GST'}, inplace=True)
        elif 'HS_Code_GST' not in df_gst_raw.columns: df_gst_raw['HS_Code_GST'] = ''

        ess_cols={'HS_Code_GST':'','Description_GST':'','CGST_Rate':'0','SGST_Rate':'0','IGST_Rate':'0','Compensation_Cess_Raw':'Nil'}
        for col,default in ess_cols.items():
            if col not in df_gst_raw.columns: df_gst_raw[col]=default
        
        df_proc = df_gst_raw[list(ess_cols.keys())].copy()

        for col in ['CGST_Rate','SGST_Rate','IGST_Rate']:
            df_proc[col] = pd.to_numeric(df_proc[col].astype(str).str.replace('%','').str.strip(),errors='coerce').fillna(0.0)
        
        def parse_cess_value(value):
            str_val = str(value).lower().strip()
            if not str_val or any(x in str_val for x in ['no', 'false', 'nil', 'exempt', '-', 'na', 'n.a.', '']):
                if re.match(r"^\d*\.?\d+$", str_val) and float(str_val) == 0: return 0.0, False
                if str_val in ['no', 'false', 'nil', 'exempt', '-', 'na', 'n.a.', '']: return 0.0, False
            match_percent = re.search(r'(\d+\.?\d*)\s*%', str_val)
            if match_percent:
                try: rate = float(match_percent.group(1)); return (rate, True) if rate > 0 else (0.0, False)
                except ValueError: pass
            match_numeric = re.search(r'(\d+\.?\d*)', str_val)
            if match_numeric:
                try: rate = float(match_numeric.group(1)); return (rate, True) if rate > 0 else (0.0, False)
                except ValueError: pass
            return 0.0, True if str_val and not any(x in str_val for x in ['no', 'false', 'nil', 'exempt', '-', 'na', 'n.a.', '']) else False

        if 'Compensation_Cess_Raw' in df_proc.columns:
            parsed_c = df_proc['Compensation_Cess_Raw'].apply(parse_cess_value)
            df_proc['Compensation_Cess_Rate'] = parsed_c.apply(lambda x: x[0])
            df_proc['Is_Compensation_Cess'] = parsed_c.apply(lambda x: x[1])
            df_proc.drop(columns=['Compensation_Cess_Raw'],inplace=True)
        else:
             df_proc['Compensation_Cess_Rate'] = 0.0
             df_proc['Is_Compensation_Cess'] = False

        df_proc['Is_Exempted']=((df_proc['CGST_Rate']==0)&(df_proc['SGST_Rate']==0)&(df_proc['IGST_Rate']==0)&((~df_proc['Is_Compensation_Cess'])|(df_proc['Compensation_Cess_Rate']==0)))
        
        df_proc.dropna(subset=['HS_Code_GST'], inplace=True)
        df_proc = df_proc[df_proc['HS_Code_GST'].astype(str).str.strip() != '']
        df_proc.drop_duplicates(subset=['HS_Code_GST'], keep='first', inplace=True)

        if df_proc.empty: print("--- GST_CSV DEBUG: CRITICAL - DataFrame is EMPTY after all CSV processing. ---")
        print(f"GST Finder: Processed {len(df_proc)} unique HSN entries from GST CSV.")
        return df_proc
    except Exception as e:
        print(f"GST Finder Error (outer in parse_gst_csv_finder): {e}")
        import traceback; traceback.print_exc(); return pd.DataFrame()

def load_all_resources():
    global doc_qa_pipeline, df_merged_gst_finder, sbert_model_gst_finder, \
           corpus_embeddings_gst_finder, qa_pipeline_gst_finder, all_descriptions_for_fuzzy_gst_finder, \
           sbert_model_item_matcher

    device_for_hf_pipeline = 0 if torch.cuda.is_available() else -1
    device_for_torch_models = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: Using device for Hugging Face Pipelines: {'CUDA' if device_for_hf_pipeline == 0 else 'CPU'}")
    print(f"INFO: Using device for PyTorch (SentenceTransformer) models: {device_for_torch_models.upper()}")

    print("INFO: Loading Document Question Answering (DocVQA) model for OCR...")
    try:
        doc_qa_pipeline = hf_pipeline(
            "document-question-answering",
            model=DOC_QA_MODEL_NAME,
            device=device_for_hf_pipeline
        )
        print(f"INFO: DocVQA model ('{DOC_QA_MODEL_NAME}') loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR loading DocVQA pipeline: {e}"); doc_qa_pipeline = None

    print(f"INFO: Loading SBERT model for Item Matching ('{SENTENCE_MODEL_NAME_ITEM_MATCHER}')...")
    try:
        sbert_model_item_matcher = SentenceTransformer(SENTENCE_MODEL_NAME_ITEM_MATCHER, device=device_for_torch_models)
        print(f"INFO: SBERT Item Matcher model loaded.")
    except Exception as e:
        print(f"ERROR loading SBERT Item Matcher: {e}"); sbert_model_item_matcher = None

    print("\nINFO: Loading GST Rate Finder resources (SBERT + QA)...")
    try:
        print("--- Load Step 1: Parsing HSN PDF ---")
        df_hsn = parse_hsn_pdf_gst_finder(None)
        print(f"--- df_hsn loaded. Shape: {df_hsn.shape if not df_hsn.empty else 'Empty'} ---")
        print("\n--- Load Step 2: Parsing GST CSV ---")
        df_gst = parse_gst_csv_gst_finder(None)
        print(f"--- df_gst loaded. Shape: {df_gst.shape if not df_gst.empty else 'Empty'} ---")
        if df_gst.empty:
            print("CRITICAL ERROR: df_gst (core GST rates) is empty. GST Finder functionality will be severely limited or non-functional.")
            df_merged_gst_finder = pd.DataFrame()
        else:
            print("\n--- Load Step 3: Merging HSN and GST data & creating Combined_Description ---")
            df_hsn_aggregated = pd.DataFrame()
            if not df_hsn.empty:
                if 'HS_Code_PDF' in df_hsn.columns: df_hsn['HS_Code_PDF'] = df_hsn['HS_Code_PDF'].astype(str).str.strip()
                df_hsn_aggregated = aggregate_hsn_descriptions_gst_finder(df_hsn.copy())
                if not df_hsn_aggregated.empty and 'HS_Code_PDF' in df_hsn_aggregated.columns:
                     df_hsn_aggregated['HS_Code_PDF'] = df_hsn_aggregated['HS_Code_PDF'].astype(str).str.strip()
            if 'HS_Code_GST' in df_gst.columns: df_gst['HS_Code_GST'] = df_gst['HS_Code_GST'].astype(str).str.strip()
            if not df_hsn_aggregated.empty:
                df_merged_gst_finder = pd.merge(df_gst, df_hsn_aggregated, left_on='HS_Code_GST', right_on='HS_Code_PDF', how='left', suffixes=('_gst', '_hsn'))
                if df_merged_gst_finder.empty and not df_gst.empty:
                    print("--- MERGE WARNING: Merge resulted in empty df, using df_gst as base. This is unusual. ---")
                    df_merged_gst_finder = df_gst.copy()
            else:
                print("--- HSN data (df_hsn_aggregated) is empty. Using df_gst as base for merged data. ---")
                df_merged_gst_finder = df_gst.copy()
            if not df_merged_gst_finder.empty:
                df_merged_gst_finder.rename(columns={'HS_Code_GST': 'HS_Code',
                                                     'Description_PDF': 'Description_PDF_from_HSN',
                                                     'Aggregated_Description_PDF': 'Aggregated_Description_PDF_from_HSN'},
                                            inplace=True, errors='ignore')
                if 'HS_Code_PDF' in df_merged_gst_finder.columns and 'HS_Code' in df_merged_gst_finder.columns and 'HS_Code_PDF' != 'HS_Code':
                    df_merged_gst_finder.drop(columns=['HS_Code_PDF'], errors='ignore', inplace=True)
                desc_gst_s = df_merged_gst_finder.get('Description_GST', pd.Series(dtype='str')).fillna('').astype(str).str.lower().str.strip()
                desc_pdf_agg_s = df_merged_gst_finder.get('Aggregated_Description_PDF_from_HSN', pd.Series(dtype='str')).fillna('').astype(str).str.lower().str.strip()
                df_merged_gst_finder['Combined_Description'] = desc_pdf_agg_s
                mask_pdf_empty = (desc_pdf_agg_s == '')
                df_merged_gst_finder.loc[mask_pdf_empty, 'Combined_Description'] = desc_gst_s[mask_pdf_empty]
                mask_both_valid_and_different = (desc_pdf_agg_s != '') & (desc_gst_s != '') & (desc_pdf_agg_s != desc_gst_s)
                df_merged_gst_finder.loc[mask_both_valid_and_different, 'Combined_Description'] = \
                    desc_pdf_agg_s[mask_both_valid_and_different] + " . " + desc_gst_s[mask_both_valid_and_different]
                df_merged_gst_finder['Combined_Description'] = df_merged_gst_finder['Combined_Description'].str.strip()
                if 'HS_Code' in df_merged_gst_finder.columns:
                     df_merged_gst_finder['HS_Code'] = df_merged_gst_finder['HS_Code'].astype(str).str.strip()
            else:
                 df_merged_gst_finder = pd.DataFrame()
        final_cols_ordered = ['HS_Code','Combined_Description','Description_GST','Description_PDF_from_HSN',
                              'Aggregated_Description_PDF_from_HSN','CGST_Rate','SGST_Rate','IGST_Rate',
                              'Is_Compensation_Cess','Compensation_Cess_Rate','Is_Exempted']
        if not df_merged_gst_finder.empty:
            for col in final_cols_ordered:
                if col not in df_merged_gst_finder.columns:
                    default_val = 0.0 if any(k in col for k in ['Rate','Cess_Rate']) else \
                                  (False if any(k in col for k in ['Is_Compensation_Cess', 'Is_Exempted']) else pd.NA)
                    df_merged_gst_finder[col] = default_val
            df_merged_gst_finder = df_merged_gst_finder[[c for c in final_cols_ordered if c in df_merged_gst_finder.columns]].copy()
        else:
            df_merged_gst_finder = pd.DataFrame(columns=final_cols_ordered)
        print(f"--- df_merged_gst_finder final shape before SBERT for GST Finder: {df_merged_gst_finder.shape} ---")
        print("\n--- Load Step 4: Preparing SBERT for GST Rate Finder ---")
        if df_merged_gst_finder.empty or 'Combined_Description' not in df_merged_gst_finder.columns or df_merged_gst_finder['Combined_Description'].isnull().all():
            print("--- CRITICAL: df_merged_gst_finder is empty or 'Combined_Description' is all null. SBERT for GST Finder cannot be initialized. ---")
            sbert_model_gst_finder = None
            corpus_embeddings_gst_finder = torch.empty(0)
            all_descriptions_for_fuzzy_gst_finder = []
        else:
            desc_for_sbert_gst = df_merged_gst_finder['Combined_Description'].fillna('').astype(str).tolist()
            desc_for_sbert_gst = [d if d.strip() else "empty" for d in desc_for_sbert_gst]
            if not any(d.strip() and d != "empty" for d in desc_for_sbert_gst):
                print("WARN: All 'Combined_Description' for GST Finder are effectively empty. SBERT embeddings will be based on 'empty' or not useful.")
                sbert_model_gst_finder = None
                corpus_embeddings_gst_finder = torch.empty(0)
            else:
                sbert_model_gst_finder = SentenceTransformer(SENTENCE_MODEL_NAME_GST_FINDER, device=device_for_torch_models)
                print(f"INFO: SBERT model ('{SENTENCE_MODEL_NAME_GST_FINDER}') for GST Finder loaded.")
                corpus_embeddings_gst_finder = sbert_model_gst_finder.encode(desc_for_sbert_gst, convert_to_tensor=True, show_progress_bar=True)
                print(f"INFO: SBERT Corpus Embeddings for GST Finder created. Shape: {corpus_embeddings_gst_finder.shape if corpus_embeddings_gst_finder is not None else 'None'}")
            all_descriptions_for_fuzzy_gst_finder = df_merged_gst_finder['Combined_Description'].astype(str).fillna('').unique().tolist()
            all_descriptions_for_fuzzy_gst_finder = [d for d in all_descriptions_for_fuzzy_gst_finder if d.strip()]
        print("\n--- Load Step 5: Loading QA Pipeline for GST Rate Finder ---")
        try:
            qa_pipeline_gst_finder = hf_pipeline(
                'question-answering',
                model=QA_MODEL_NAME_GST_FINDER,
                tokenizer=QA_MODEL_NAME_GST_FINDER,
                device=device_for_hf_pipeline
            )
            print(f"INFO: QA pipeline ('{QA_MODEL_NAME_GST_FINDER}') for GST Finder loaded.")
        except Exception as e:
            print(f"ERROR loading QA pipeline for GST Finder: {e}"); qa_pipeline_gst_finder = None
    except Exception as e:
        print(f"CRITICAL ERROR during GST Rate Finder data/model loading: {e}")
        import traceback; traceback.print_exc()
        df_merged_gst_finder = pd.DataFrame()
        sbert_model_gst_finder = None
        corpus_embeddings_gst_finder = torch.empty(0)
        qa_pipeline_gst_finder = None
        all_descriptions_for_fuzzy_gst_finder = []


def extract_invoice_data(file_path):
    global doc_qa_pipeline, PDF2IMAGE_AVAILABLE, PYTESSERACT_AVAILABLE
    extracted_data = {key.replace('_raw',''): None for key in QUESTIONS_FOR_OCR.keys()}
    if not doc_qa_pipeline:
        print("ERROR: DocVQA OCR pipeline is not loaded in extract_invoice_data.")
        return {"error": "OCR Service Unavailable. DocVQA pipeline not loaded."}
    file_extension = ""
    if '.' in os.path.basename(file_path):
        file_extension = os.path.basename(file_path).rsplit('.', 1)[1].lower()
    pil_image_to_process = None
    try:
        if file_extension == 'pdf':
            if not PDF2IMAGE_AVAILABLE:
                return {"error": "PDF processing library (pdf2image) not available."}
            if not PYTESSERACT_AVAILABLE:
                 print("WARNING: Pytesseract/Tesseract not available. DocVQA may struggle with PDF text extraction if it needs to OCR from scratch.")
            try:
                print(f"INFO: Converting PDF '{os.path.basename(file_path)}' (first page) for DocVQA...")
                images_from_pdf = convert_from_path(file_path, first_page=1, last_page=1, dpi=200, timeout=30)
                if not images_from_pdf:
                    return {"error": "PDF to image conversion failed (no images returned from pdf2image)."}
                pil_image_to_process = images_from_pdf[0].convert("RGB")
            except Exception as e:
                err_msg = f"PDF to image conversion error: {str(e)}. Ensure Poppler is installed and accessible."
                print(f"ERROR: {err_msg}")
                return {"error": err_msg}
        elif file_extension in {'png', 'jpg', 'jpeg'}:
            pil_image_to_process = Image.open(file_path).convert("RGB")
        else:
            return {"error": f"Unsupported file type for OCR: '{file_extension}'"}
        if pil_image_to_process is None:
            return {"error": "Image preparation for OCR failed."}
        for question_key, question_text in QUESTIONS_FOR_OCR.items():
            mapped_key = question_key.replace('_raw','')
            try:
                qa_results_list = doc_qa_pipeline(image=pil_image_to_process, question=question_text)
                current_answer = None
                if isinstance(qa_results_list, list) and qa_results_list:
                    answer_obj = qa_results_list[0]
                    current_answer = answer_obj.get('answer')
                elif isinstance(qa_results_list, dict) and 'answer' in qa_results_list:
                    current_answer = qa_results_list['answer']
                extracted_data[mapped_key] = str(current_answer).strip() if current_answer and isinstance(current_answer, str) else current_answer
            except Exception as e:
                error_detail = str(e)
                if "pytesseract" in error_detail.lower() or "tesseract" in error_detail.lower():
                     error_msg_display = "OCR engine (Tesseract) error. Ensure it's installed & configured."
                     print(f"ERROR: Tesseract related error during DocVQA for '{question_key}': {error_detail}")
                else:
                     error_msg_display = f"QA error for '{question_key}'"
                extracted_data[mapped_key] = f"OCR Error: {error_msg_display}"
        return extracted_data
    except Exception as e:
        import traceback
        print(f"FATAL OCR ERROR in extract_invoice_data for '{os.path.basename(file_path)}':")
        traceback.print_exc()
        return {"error": f"Critical OCR system error: {str(e)}"}


def get_gst_rates_for_product_page_query(user_query):
    global df_merged_gst_finder, sbert_model_gst_finder, corpus_embeddings_gst_finder, qa_pipeline_gst_finder, all_descriptions_for_fuzzy_gst_finder
    if df_merged_gst_finder.empty: return [{"error": "GST data not loaded."}]
    if not (sbert_model_gst_finder and qa_pipeline_gst_finder and
            corpus_embeddings_gst_finder is not None and corpus_embeddings_gst_finder.nelement() > 0):
        return [{"error": "GST Rate Finder models (SBERT/QA for GST data) not fully initialized."}]
    if not user_query or not str(user_query).strip(): return [{"error": "Query is empty."}]
    query_cleaned = str(user_query).lower().strip();
    retrieved_item_info = None
    if re.fullmatch(r'\d{2,8}', query_cleaned):
        df_merged_gst_finder['HS_Code_str'] = df_merged_gst_finder['HS_Code'].astype(str).str.strip()
        exact_hs_match_df = df_merged_gst_finder[df_merged_gst_finder['HS_Code_str'] == query_cleaned]
        if not exact_hs_match_df.empty:
            retrieved_item_info = {'row': exact_hs_match_df.iloc[0].copy(), 'score': 1.0, 'match_type': 'Exact HS Code'}
        df_merged_gst_finder.drop(columns=['HS_Code_str'], inplace=True, errors='ignore')
    if retrieved_item_info is None:
        try:
            q_emb = sbert_model_gst_finder.encode(query_cleaned, convert_to_tensor=True)
            cos_scores = sbert_util.cos_sim(q_emb, corpus_embeddings_gst_finder)[0]
            if len(cos_scores) > 0:
                top_sbert_res = torch.topk(cos_scores, k=1)
                s_score, s_idx = top_sbert_res.values[0].item(), top_sbert_res.indices[0].item()
                s_thresh = 0.35
                if s_score >= s_thresh:
                    retrieved_item_info = {'row': df_merged_gst_finder.iloc[s_idx].copy(), 'score': s_score, 'match_type': 'SBERT (GST Rate)'}
        except Exception as e:
            print(f"ERROR (Page Query): SBERT search for GST rates failed: {e}")
    if retrieved_item_info is None and all_descriptions_for_fuzzy_gst_finder:
        best_f = fuzzy_process.extractOne(query_cleaned, all_descriptions_for_fuzzy_gst_finder)
        if best_f:
            f_desc, f_score_100 = best_f[0], best_f[1]
            f_score = f_score_100 / 100.0
            f_thresh = 0.75
            if f_score >= f_thresh:
                match_rows_df = df_merged_gst_finder[df_merged_gst_finder['Combined_Description'] == f_desc]
                if not match_rows_df.empty:
                    retrieved_item_info = {'row': match_rows_df.iloc[0].copy(), 'score': f_score, 'match_type': 'Fuzzy (GST Rate)'}
    if retrieved_item_info is None:
        return [{"error": f"No relevant items found for '{user_query}' in GST database."}]
    best_match_row = retrieved_item_info['row']
    match_score = retrieved_item_info['score']
    match_type = retrieved_item_info['match_type']
    retrieved_hs_code_specific = str(best_match_row.get('HS_Code', 'N/A')).strip()
    cgst_db = best_match_row.get('CGST_Rate', 0.0)
    sgst_db = best_match_row.get('SGST_Rate', 0.0)
    igst_db = best_match_row.get('IGST_Rate', 0.0)
    is_exempted_db = bool(best_match_row.get('Is_Exempted', False))
    cess_applicable_db = bool(best_match_row.get('Is_Compensation_Cess', False))
    cess_rate_db = best_match_row.get('Compensation_Cess_Rate', 0.0)
    final_cgst_str = f"{cgst_db}%" if cgst_db is not None else "N/A"
    final_sgst_str = f"{sgst_db}%" if sgst_db is not None else "N/A"
    final_igst_str = f"{igst_db}%" if igst_db is not None else "N/A"
    final_cess_app_str = "Yes" if cess_applicable_db else "No"
    final_cess_rate_str = f"{cess_rate_db}%" if cess_applicable_db else "N/A"
    final_exempt_str = "Yes" if is_exempted_db else "No"

    res_template = [{
        "hs_code_db": retrieved_hs_code_specific,
        "cgst_rate": final_cgst_str,
        "sgst_rate": final_sgst_str,
        "igst_rate": final_igst_str,
        "cess_applicable": final_cess_app_str,
        "cess_rate": final_cess_rate_str,
        "is_exempted": final_exempt_str,
        "retrieval_score": f"{match_score:.2f} ({match_type})",
    }]
    return res_template


@app.route('/')
def home(): return render_template('index.html')

@app.route('/reconcile', methods=['GET', 'POST'])
def reconcile_page():
    if request.method == 'POST':
        if 'invoice1' not in request.files or 'invoice2' not in request.files:
            flash('Both invoice files are required!', 'error'); return redirect(request.url)
        file1, file2 = request.files['invoice1'], request.files['invoice2']
        if not file1.filename or not file2.filename :
            flash('One or both files not selected!', 'error'); return redirect(request.url)
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            fname1_orig, fname2_orig = secure_filename(file1.filename), secure_filename(file2.filename)
            uid = uuid.uuid4().hex[:8]
            ext1 = fname1_orig.rsplit('.',1)[1].lower() if '.' in fname1_orig else 'dat'
            ext2 = fname2_orig.rsplit('.',1)[1].lower() if '.' in fname2_orig else 'dat'
            if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
            fpath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_inv1.{ext1}")
            fpath2 = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_inv2.{ext2}")
            try:
                file1.save(fpath1); file2.save(fpath2)
                flash(f'Files "{fname1_orig}" & "{fname2_orig}" uploaded. Processing...', 'info')
                data1_ocr = extract_invoice_data(fpath1)
                data2_ocr = extract_invoice_data(fpath2)
                ocr_error_messages = []
                if data1_ocr.get("error"): ocr_error_messages.append(f"Invoice 1 OCR: {data1_ocr['error']}")
                if data2_ocr.get("error"): ocr_error_messages.append(f"Invoice 2 OCR: {data2_ocr['error']}")
                if ocr_error_messages:
                    flash(". ".join(ocr_error_messages), 'error')
                engine = GSTReconciliationEngine()
                comp_res = engine.compare_invoices(data1_ocr, data2_ocr)
                report_html = engine.generate_reconciliation_report(comp_res, inv1_filename=fname1_orig, inv2_filename=fname2_orig)
                return render_template('reconciliation_report_display.html', report_content=report_html)
            except Exception as e:
                flash(f'An unexpected error occurred during processing: {e}', 'error')
                import traceback; traceback.print_exc()
                return redirect(request.url)
        else:
            flash(f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
            return redirect(request.url)
    return render_template('reconcile.html')

@app.route('/find-gst-rate', methods=['GET', 'POST'])
def find_gst_rate_page():
    results, query_val = None, ""
    if request.method == 'POST':
        query_val = request.form.get('query','').strip()
        if query_val:
            if df_merged_gst_finder.empty or not (sbert_model_gst_finder and qa_pipeline_gst_finder and
                                                 corpus_embeddings_gst_finder is not None and
                                                 corpus_embeddings_gst_finder.nelement() > 0):
                 results = [{"error":"GST Rate Finder backend not fully initialized or data is missing. Check server logs."}]
            else:
                results = get_gst_rates_for_product_page_query(query_val)
        else:
            results = [{"error":"Please enter a product description or HS code."}]
    return render_template('find_gst_rate.html', results=results, query=query_val)

@app.route('/uploads/<filename>')
def uploaded_file(filename): return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER); print(f"INFO: Created '{UPLOAD_FOLDER}'")
    print("INFO: Starting application, loading all resources. This may take a few minutes...")
    load_all_resources()
    print("\n--- Resource Loading Sanity Check ---")
    if doc_qa_pipeline is None: print("CRITICAL WARNING: DocVQA OCR pipeline FAILED to load. Invoice OCR WILL NOT WORK.")
    else: print(f"INFO: DocVQA OCR pipeline ('{DOC_QA_MODEL_NAME}') loaded.")
    if sbert_model_item_matcher is None: print("WARNING: SBERT Item Matcher model FAILED to load. Line item matching quality will be lower (fallback to basic).")
    else: print(f"INFO: SBERT Item Matcher ('{SENTENCE_MODEL_NAME_ITEM_MATCHER}') loaded.")
    if df_merged_gst_finder.empty: print("CRITICAL WARNING: GST Rate Finder data (df_merged_gst_finder) is EMPTY. GST lookup will not work.")
    if sbert_model_gst_finder is None : print("WARNING: SBERT model for GST Rate Finder FAILED to load.")
    if corpus_embeddings_gst_finder is None or \
       (isinstance(corpus_embeddings_gst_finder, torch.Tensor) and corpus_embeddings_gst_finder.nelement() == 0):
        print("WARNING: SBERT corpus embeddings for GST Rate Finder are EMPTY or FAILED to load.")
    if qa_pipeline_gst_finder is None: print("WARNING: QA pipeline for GST Rate Finder FAILED to load.")
    if not df_merged_gst_finder.empty and sbert_model_gst_finder and qa_pipeline_gst_finder and \
       (corpus_embeddings_gst_finder is not None and corpus_embeddings_gst_finder.nelement() > 0):
        print("INFO: Core GST Rate Finder resources (Data, SBERT, QA) appear loaded successfully.")
    else: print("ERROR: One or more critical resources for GST Rate Finder FAILED to load or are incomplete.")
    print("--- End Sanity Check ---")
    if not PYTESSERACT_AVAILABLE:
        print("\nIMPORTANT SYSTEM WARNING: 'pytesseract' library not found or Tesseract OCR engine not installed/in PATH.")
        print("The DocVQA pipeline's ability to OCR images/PDFs from scratch will be compromised or fail.")
        print("Please ensure Tesseract is correctly installed and configured for optimal performance.")
    if not PDF2IMAGE_AVAILABLE:
        print("\nCRITICAL SYSTEM WARNING: 'pdf2image' library or its Poppler dependency is not installed/configured.")
        print("Processing PDF files for OCR WILL FAIL. Please install and configure them.")
    print("\nINFO: Application setup complete. Flask server starting...")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))