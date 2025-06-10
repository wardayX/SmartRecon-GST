# app.py (Combined - SBERT + QA for GST Finder, DONUT for Reconcile - Mimicking Notebook's SBERT approach)
import os
import io
import re
import json
import uuid
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

import pandas as pd
import pdfplumber
from fuzzywuzzy import process as fuzzy_process
from sentence_transformers import SentenceTransformer, util as sbert_util # Using SBERT again
import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    pipeline as hf_pipeline
)
# TF-IDF related imports are no longer needed for the primary GST Finder path
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

HSN_PDF_PATH = r"HSN-Codes-for-GST-Enrolment.pdf" # ENSURE THIS FILE EXISTS
GST_CSV_PATH = r"cbic_gst_goods_rates_exact.csv"   # ENSURE THIS FILE EXISTS
SENTENCE_MODEL_NAME_GST_FINDER = 'all-mpnet-base-v2' # SBERT model
QA_MODEL_NAME_GST_FINDER = 'deepset/roberta-base-squad2'
DONUT_MODEL_NAME = "philschmid/donut-base-sroie"

# --- Global Variables ---
# For GST Rate Finder (SBERT approach)
df_merged_gst_finder = pd.DataFrame()
sbert_model_gst_finder = None         # For SBERT (like notebook's 'model')
corpus_embeddings_gst_finder = None # For SBERT (like notebook's 'corpus_embeddings')
qa_pipeline_gst_finder = None         # For QA (like notebook's 'qa_pipeline')
all_descriptions_for_fuzzy_gst_finder = []

# For Invoice Reconciliation
donut_processor = None
donut_model = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a_different_very_secret_key_please_change' # Change this
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class GSTReconciliationEngine:
    # ... (All GSTReconciliationEngine methods EXCEPT _get_expected_gst_profile_for_item
    #      remain IDENTICAL to the last fully working version where UnboundLocalError was fixed.
    #      For brevity, I will only show the changed _get_expected_gst_profile_for_item and its caller.)
    def __init__(self, tolerance=0.01):
        self.tolerance = tolerance

    def _parse_float(self, value, default=0.0):
        if value is None: return default
        try:
            cleaned_value = str(value).replace('%', '')
            cleaned_value = re.sub(r'[^\d\.\-]', '', cleaned_value)
            return float(cleaned_value) if cleaned_value and cleaned_value != '-' else default
        except (ValueError, TypeError): return default

    def validate_gstin(self, gstin):
        if not gstin or len(str(gstin)) != 15: return False
        pattern = r'^[0-9]{2}[A-Z0-9]{10}[0-9][A-Z][0-9A-Z]$'
        return bool(re.match(pattern, str(gstin)))

    def map_donut_to_reconciler_format(self, donut_output): # Adapted for SROIE
        if not isinstance(donut_output, dict): return {}
        mapped = {}
        mapped['invoice_number'] = donut_output.get('id', donut_output.get('doc_id', None)) 
        mapped['invoice_date'] = donut_output.get('date', None)
        mapped['supplier_name'] = donut_output.get('company', donut_output.get('store_name', None))
        mapped['supplier_address'] = donut_output.get('address', None)
        mapped['supplier_gstin'] = donut_output.get('vat_no', None) 
        mapped['buyer_name'] = None
        mapped['buyer_gstin'] = None
        mapped['subtotal'] = self._parse_float(donut_output.get('sub_total', donut_output.get('subtotal', 0)))
        mapped['total_gst'] = self._parse_float(donut_output.get('tax', 0))
        mapped['grand_total'] = self._parse_float(donut_output.get('total', 0))
        donut_menu = donut_output.get('menu', {})
        item_names = donut_menu.get('nm', [])
        item_prices = donut_menu.get('price', []) 
        item_quantities = donut_menu.get('cnt', [])
        mapped_items = []
        if isinstance(item_names, list):
            for i, name in enumerate(item_names):
                qty_str = item_quantities[i] if i < len(item_quantities) else '1'
                price_str = item_prices[i] if i < len(item_prices) else '0'
                quantity = self._parse_float(qty_str, 1)
                total_price_for_item_line = self._parse_float(price_str, 0)
                unit_price = (total_price_for_item_line / quantity) if quantity > 0 else total_price_for_item_line
                mapped_items.append({'description': name.strip() if name else "Unknown Item", 'quantity': quantity, 'rate': unit_price, 
                                     'amount': total_price_for_item_line, 'gst_rate': 0.0, 'gst_amount': 0.0, 
                                     'total_amount': total_price_for_item_line })
        mapped['items'] = mapped_items
        return mapped

    def _compare_basic_fields(self, inv1, inv2, result):
        basic_fields = ['invoice_number', 'invoice_date', 'supplier_name', 'supplier_gstin', 'buyer_name', 'buyer_gstin']
        for field in basic_fields:
            val1, val2 = str(inv1.get(field, '')).strip().upper(), str(inv2.get(field, '')).strip().upper()
            if val1 is None: val1 = ""
            if val2 is None: val2 = ""
            if val1 == val2 and val1 != "": result['matches'][field] = {'value': val1, 'status': 'MATCH'}
            elif val1 != val2: result['mismatches'][field] = {'invoice1': val1 or "N/A", 'invoice2': val2 or "N/A", 'status': 'MISMATCH'}

    def _compare_gst_details(self, inv1, inv2, result):
        for field in ['supplier_gstin', 'buyer_gstin']:
            g1, g2 = inv1.get(field), inv2.get(field)
            result['gst_calculations'][f'{field}_validation'] = {'invoice1_valid': self.validate_gstin(g1) if g1 else False,
                                                              'invoice2_valid': self.validate_gstin(g2) if g2 else False,
                                                              'gstin1': g1 or "N/A", 'gstin2': g2 or "N/A"}
    def _reconcile_amounts(self, inv1, inv2, result):
        for field in ['subtotal', 'total_gst', 'grand_total']:
            a1,a2 = self._parse_float(inv1.get(field,0)), self._parse_float(inv2.get(field,0))
            diff = abs(a1-a2)
            if a1==0 and a2==0: continue
            if diff <= self.tolerance: result['matches'][field] = {'invoice1':a1,'invoice2':a2,'difference':diff,'status':'MATCH'}
            else: result['mismatches'][field] = {'invoice1':a1,'invoice2':a2,'difference':diff,'status':'MISMATCH'}

    def _reconcile_line_items(self, items1, items2, result):
        result['item_reconciliation'] = {'total_items_inv1':len(items1),'total_items_inv2':len(items2),'item_matches':[],'unmatched_items':[]}
        matched_idx2 = set()
        for i1, item1 in enumerate(items1):
            desc1 = str(item1.get('description','')).lower().strip()
            if not desc1: continue
            best_match_idx2, best_score = -1, 0.0
            for i2, item2 in enumerate(items2):
                if i2 in matched_idx2: continue
                desc2 = str(item2.get('description','')).lower().strip()
                if not desc2: continue
                score = fuzzy_process.extractOne(desc1, [desc2])[1]/100.0
                if score > best_score: best_score, best_match_idx2 = score, i2
            if best_match_idx2 != -1 and best_score >= 0.70: 
                matched_idx2.add(best_match_idx2)
                comp = self._compare_single_item(item1, items2[best_match_idx2])
                result['item_reconciliation']['item_matches'].append({'item1':item1,'item2':items2[best_match_idx2],'comparison':comp,'score':best_score})
            else: result['item_reconciliation']['unmatched_items'].append({'source':'invoice1','item':item1})
        for i2, item2 in enumerate(items2):
            if i2 not in matched_idx2: result['item_reconciliation']['unmatched_items'].append({'source':'invoice2','item':item2})

    def _compare_single_item(self, item1, item2):
        comp = {'matches':{},'mismatches':{}}
        for field in ['quantity','rate','amount','total_amount']:
            v1,v2 = self._parse_float(item1.get(field,0)), self._parse_float(item2.get(field,0))
            if v1==0 and v2==0: continue
            diff = abs(v1-v2)
            tol = self.tolerance * max(v1,v2) if 'amount' in field or 'rate' in field else self.tolerance
            if diff <= tol: comp['matches'][field]={'value1':v1,'value2':v2,'difference':diff}
            else: comp['mismatches'][field]={'value1':v1,'value2':v2,'difference':diff}
        return comp

    def _get_expected_gst_profile_for_item(self, description): # REVERTED TO SBERT logic
        global df_merged_gst_finder, sbert_model_gst_finder, corpus_embeddings_gst_finder, qa_pipeline_gst_finder
        
        # Basic check for readiness of resources
        if df_merged_gst_finder.empty: return {"error": "GST Finder: DB not loaded.", "db_description": description}
        if sbert_model_gst_finder is None: return {"error": "GST Finder: Semantic model not loaded.", "db_description": description}
        if corpus_embeddings_gst_finder is None or corpus_embeddings_gst_finder.nelement() == 0:
             return {"error": "GST Finder: Corpus embeddings not available.", "db_description": description}
        if qa_pipeline_gst_finder is None: return {"error": "GST Finder: QA model not loaded.", "db_description": description}
        if not description or not str(description).strip(): return {"error": "Please provide product description.", "db_description": description}

        try:
            query_embedding = sbert_model_gst_finder.encode(str(description).lower().strip(), convert_to_tensor=True)
            cos_scores = sbert_util.cos_sim(query_embedding, corpus_embeddings_gst_finder)[0]
            
            if len(cos_scores) == 0: return {"error": "Internal error: No corpus scores.", "db_description": description}

            top_result_idx = torch.topk(cos_scores, k=1).indices.tolist()[0]
            sbert_score = cos_scores[top_result_idx].item()

            sbert_relevance_threshold = 0.55 # Keep this as tuned before
            if sbert_score < sbert_relevance_threshold:
                error_msg = (f"Could not find a confident GST rate match for '{description}'. "
                             f"(Best SBERT score {sbert_score:.2f} < threshold {sbert_relevance_threshold}).")
                return {"error": error_msg, "db_description": description, "best_match_score_debug": sbert_score}

            row_from_db = df_merged_gst_finder.iloc[top_result_idx]
            db_desc_match = row_from_db.get('Combined_Description', 'N/A')
            hs_match = row_from_db.get('HS_Code', row_from_db.get('HS_Code_GST', 'N/A'))

            context = (f"The product is '{db_desc_match}' with HS Code {hs_match}. "
                       f"CGST: {row_from_db.get('CGST_Rate', 'unknown')}%. "
                       f"SGST/UTGST: {row_from_db.get('SGST_Rate', 'unknown')}%. "
                       f"IGST: {row_from_db.get('IGST_Rate', 'unknown')}%. "
                       f"Compensation cess is {'applicable' if row_from_db.get('Is_Compensation_Cess') else 'not applicable'}. "
                       f"Cess rate is {row_from_db.get('Compensation_Cess_Rate', '0')}%. "
                       f"Exempted: {'yes' if row_from_db.get('Is_Exempted') else 'no'}.")
            
            qa_results = {}
            questions_map = {"igst_qa": "What is the IGST rate?", "cgst_qa": "What is the CGST rate?",
                             "sgst_qa": "What is the SGST rate?", "is_exempted_qa": "Is the product exempted from tax?"}
            for key, q_text in questions_map.items():
                ans = qa_pipeline_gst_finder(question=q_text, context=context)
                answer_text, num_match = ans['answer'], re.search(r'(\d+\.?\d*)', ans['answer'])
                qa_results[key] = num_match.group(1) if num_match else answer_text
            
            return {"db_description":db_desc_match, "hs_code_db":hs_match, "score_db":sbert_score, # score_db is now SBERT score
                    "igst_db":self._parse_float(qa_results.get("igst_qa","0")), "cgst_db":self._parse_float(qa_results.get("cgst_qa","0")),
                    "sgst_db":self._parse_float(qa_results.get("sgst_qa","0")),
                    "is_exempted_db":any(kw in str(qa_results.get("is_exempted_qa","no")).lower() for kw in ["yes","exempted"])}
        except Exception as e:
            print(f"Error in _get_expected_gst_profile_for_item (SBERT path) for '{description}': {e}")
            import traceback; traceback.print_exc()
            return {"error":f"Internal error during SBERT/QA lookup: {str(e)}","db_description":description}

    def verify_item_gst_rates(self, item_from_invoice):
        # This method calls _get_expected_gst_profile_for_item, which now uses SBERT
        item_desc_on_invoice = item_from_invoice.get('description', '')
        invoice_item_gst_rate = self._parse_float(item_from_invoice.get('gst_rate', 0.0))
        expected_profile = self._get_expected_gst_profile_for_item(item_desc_on_invoice)
        status_details = {"invoice_item_description": item_desc_on_invoice,
                          "invoice_rate_detected": invoice_item_gst_rate if invoice_item_gst_rate > 0 else "N/A (or 0%)",
                          "db_match_description": expected_profile.get("db_description", "N/A"),
                          "db_match_score": f"{expected_profile.get('score_db', 0):.2f}" if "score_db" in expected_profile else "N/A",
                          "expected_igst_db": expected_profile.get("igst_db", "N/A"), "expected_cgst_db": expected_profile.get("cgst_db", "N/A"),
                          "expected_sgst_db": expected_profile.get("sgst_db", "N/A"), "is_exempted_db": expected_profile.get("is_exempted_db", "Unknown"),
                          "status": "UNVERIFIED"}
        if "error" in expected_profile:
            status_details["status"] = f"DB_LOOKUP_FAILED ({expected_profile['error']})"
            return status_details
        db_igst, db_cgst, db_sgst = expected_profile.get("igst_db",0.0), expected_profile.get("cgst_db",0.0), expected_profile.get("sgst_db",0.0)
        effective_db_rate = db_igst if db_igst > 0 else (db_cgst + db_sgst)
        if expected_profile.get("is_exempted_db", False): effective_db_rate = 0.0
        status_details["effective_expected_rate_db"] = effective_db_rate
        if isinstance(invoice_item_gst_rate, float) and invoice_item_gst_rate > 0.0:
            status_details["status"] = "AUTHENTIC" if abs(invoice_item_gst_rate - effective_db_rate) <= 0.1 else "RATE_MISMATCH"
        elif expected_profile.get("is_exempted_db", False) and (invoice_item_gst_rate == 0.0 or invoice_item_gst_rate == "N/A (or 0%)"):
            status_details["status"] = "AUTHENTIC (Exempted)"
        elif isinstance(invoice_item_gst_rate, float) and invoice_item_gst_rate == 0.0 and not expected_profile.get("is_exempted_db", False) and effective_db_rate > 0:
            status_details["status"] = "RATE_MISMATCH (Invoice 0%, DB expects rate)"
        else: status_details["status"] = "UNVERIFIED (Invoice Rate Missing/Zero & DB Not Exempt)"
        return status_details
        
    def compare_invoices(self, invoice1_data_orig_donut, invoice2_data_orig_donut, comparison_type="buyer_seller_ocr"):
        # ... (This method remains the same) ...
        result = {'comparison_id': f"CMP_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}", 'comparison_type': comparison_type,
                  'invoice1_orig_data': invoice1_data_orig_donut, 'invoice2_orig_data': invoice2_data_orig_donut,
                  'invoice1_mapped': self.map_donut_to_reconciler_format(invoice1_data_orig_donut),
                  'invoice2_mapped': self.map_donut_to_reconciler_format(invoice2_data_orig_donut),
                  'matches': {}, 'mismatches': {}, 'gst_calculations': {}, 'item_reconciliation': {},
                  'item_gst_authenticity_inv1': [], 'overall_status': 'UNKNOWN', 'confidence_score': 0.0, 'summary': {}}
        self._compare_basic_fields(result['invoice1_mapped'], result['invoice2_mapped'], result)
        self._compare_gst_details(result['invoice1_mapped'], result['invoice2_mapped'], result)
        self._reconcile_amounts(result['invoice1_mapped'], result['invoice2_mapped'], result)
        items1_m, items2_m = result['invoice1_mapped'].get('items', []), result['invoice2_mapped'].get('items', [])
        if items1_m or items2_m: self._reconcile_line_items(items1_m, items2_m, result)
        for item_inv1 in items1_m:
            result['item_gst_authenticity_inv1'].append(self.verify_item_gst_rates(item_inv1))
        self._calculate_overall_status(result)
        self._generate_summary(result)
        return result

    def _calculate_overall_status(self, result):
        # ... (This method remains the same) ...
        basic_match_count, basic_mismatch_count = len(result['matches']), len(result['mismatches'])
        total_basic_checks = basic_match_count + basic_mismatch_count
        if total_basic_checks == 0:
            result['overall_status'], result['confidence_score'] = 'NO_DATA_FOR_COMPARISON', 0.0; return
        base_confidence = (basic_match_count / total_basic_checks) * 100
        item_rec_score = 0.0
        if 'item_reconciliation' in result and result['item_reconciliation'].get('item_matches'):
            total_items = max(1,result['item_reconciliation']['total_items_inv1'],result['item_reconciliation']['total_items_inv2'])
            item_rec_score = (len(result['item_reconciliation']['item_matches']) / total_items) * 100
            base_confidence = (base_confidence * 0.7) + (item_rec_score * 0.3)
        gst_auth_score = 0.0
        if result.get('item_gst_authenticity_inv1'):
            auth_items = result['item_gst_authenticity_inv1']
            authentic_count = sum(1 for item in auth_items if "AUTHENTIC" in item.get('status', ''))
            verifiable_count = sum(1 for item in auth_items if "UNVERIFIED" not in item.get('status','') and "DB_LOOKUP_FAILED" not in item.get('status',''))
            if verifiable_count > 0:
                gst_auth_score = (authentic_count / verifiable_count) * 100
                base_confidence = (base_confidence * 0.6) + (gst_auth_score * 0.4)
            elif auth_items: base_confidence *= 0.8
        result['confidence_score'] = round(base_confidence, 1)
        status_conditions = (len(result['mismatches']) == 0, item_rec_score >= 90, (gst_auth_score >= 90 if result.get('item_gst_authenticity_inv1') else True))
        if all(status_conditions): result['overall_status'] = 'PERFECT_MATCH'
        elif result['confidence_score'] >= 75: result['overall_status'] = 'GOOD_MATCH'
        elif result['confidence_score'] >= 50: result['overall_status'] = 'PARTIAL_MATCH'
        else: result['overall_status'] = 'POOR_MATCH'
        critical_mismatch = ('grand_total' in result['mismatches'] or \
            ('supplier_gstin' in result['mismatches'] and result['invoice1_mapped'].get('supplier_gstin')) or \
            ('buyer_gstin' in result['mismatches'] and result['invoice1_mapped'].get('buyer_gstin')))
        if critical_mismatch:
            if result['overall_status'] in ['PERFECT_MATCH', 'GOOD_MATCH']: result['overall_status'] = 'PARTIAL_MATCH (Critical Field Mismatch)'
            elif result['overall_status'] == 'PARTIAL_MATCH': result['overall_status'] = 'POOR_MATCH (Critical Field Mismatch)'

    def _generate_summary(self, result):
        # ... (This method remains the same) ...
        result['summary'] = {'total_fields_compared': len(result['matches'])+len(result['mismatches']), 'fields_matched': len(result['matches']),
                             'fields_mismatched': len(result['mismatches']), 'amount_differences': [], 'critical_mismatches': []}
        for field, data in result['mismatches'].items():
            if 'difference' in data and isinstance(data['difference'], (int, float)):
                result['summary']['amount_differences'].append({'field': field, 'difference': data['difference']})
        critical_fields = ['grand_total', 'supplier_gstin', 'buyer_gstin']
        for field in critical_fields:
            if field in result['mismatches'] and (result['invoice1_mapped'].get(field) or result['invoice2_mapped'].get(field)):
                result['summary']['critical_mismatches'].append(field)

    def get_html_template(self):
        # ... (This method remains the same, containing the full HTML string with placeholders) ...
        return """<!DOCTYPE html><html><head><title>Reconciliation Report</title><meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:0;padding:0;background-color:#f4f7f6;}.report-container{max-width:1200px;margin:20px auto;background:white;padding:25px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08);}h1,h2,h3{color:#333;}h1{text-align:center;color:#1a535c;margin-bottom:5px;}.report-meta{text-align:center;font-size:0.9em;color:#555;margin-bottom:20px;}table{width:100%;border-collapse:collapse;margin-bottom:25px;font-size:0.9em;}th,td{border:1px solid #e0e0e0;padding:10px 12px;text-align:left;vertical-align:top;}th{background-color:#f0f4f8;color:#334e68;font-weight:600;}tr:nth-child(even){background-color:#fbfcfc;}.match{color:#27ae60;font-weight:bold;}.mismatch{color:#e74c3c;font-weight:bold;}.status-badge{display:inline-block;padding:6px 12px;border-radius:15px;color:white;font-weight:bold;font-size:1em;margin-bottom:15px;}.status-perfect_match{background-color:#27ae60;}.status-good_match{background-color:#f39c12;}.status-partial_match{background-color:#e67e22;}.status-poor_match{background-color:#c0392b;}.status-no_data_for_comparison{background-color:#7f8c8d;}.invoice-details-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}.invoice-card{background:#f9f9f9;padding:15px;border-radius:5px;border:1px solid #e8e8e8;}.invoice-card h3{margin-top:0;color:#34495e;font-size:1.1em;border-bottom:1px solid #ddd;padding-bottom:5px;margin-bottom:10px;}.summary-card{background-color:#e9f5ff;border-left:4px solid #3498db;padding:12px;margin-bottom:8px;border-radius:4px;font-size:0.95em;}.status-authentic{color:#27ae60;font-weight:bold;}.status-authentic-exempted{color:#27ae60;}.status-rate_mismatch{color:#e74c3c;font-weight:bold;}.status-db_lookup_failed{color:#f39c12;}.status-unverified-invoice-rate-missing-zero-db-not-exempt{color:#7f8c8d;}.status-unverified{color:#95a5a6;}@media (max-width:768px){.invoice-details-grid{grid-template-columns:1fr;}th,td{padding:8px;}}</style></head>
<body><div class="report-container"><h1>GST Reconciliation Report</h1>
<p class="report-meta">Comparison ID: <strong>{{COMPARISON_ID}}</strong> | Generated: {{GENERATED_DATE}}</p>
<div style="text-align:center;"><span class="status-badge status-{{OVERALL_STATUS_CLASS}}">Overall Status: {{OVERALL_STATUS}} ({{CONFIDENCE_SCORE}}% Match Score)</span></div>
<h2>Summary</h2><div class="summary-card">Fields Compared: {{TOTAL_FIELDS}}, Matched: {{FIELDS_MATCHED}}, Mismatched: {{FIELDS_MISMATCHED}}</div>
<div class="summary-card">Comparison Type: {{COMPARISON_TYPE}}</div>
{% if CRITICAL_MISMATCHES %}<div class="summary-card" style="background-color:#ffebee;border-left-color:#e74c3c;">Critical Mismatches found in: {{CRITICAL_MISMATCHES_STR}}</div>{% endif %}
<h2>Invoice Details</h2><div class="invoice-details-grid">
<div class="invoice-card"><h3>Invoice 1 ({{INV1_FILENAME}})</h3><p><strong>Inv No:</strong> {{INV1_NUMBER}} | <strong>Date:</strong> {{INV1_DATE}}</p><p><strong>Supplier:</strong> {{INV1_SUPPLIER_NAME}} (GSTIN: {{INV1_SUPPLIER_GSTIN}})</p><p><strong>Buyer:</strong> {{INV1_BUYER_NAME}} (GSTIN: {{INV1_BUYER_GSTIN}})</p><p><strong>Grand Total:</strong> ₹{{INV1_GRAND_TOTAL}}</p></div>
<div class="invoice-card"><h3>Invoice 2 ({{INV2_FILENAME}})</h3><p><strong>Inv No:</strong> {{INV2_NUMBER}} | <strong>Date:</strong> {{INV2_DATE}}</p><p><strong>Supplier:</strong> {{INV2_SUPPLIER_NAME}} (GSTIN: {{INV2_SUPPLIER_GSTIN}})</p><p><strong>Buyer:</strong> {{INV2_BUYER_NAME}} (GSTIN: {{INV2_BUYER_GSTIN}})</p><p><strong>Grand Total:</strong> ₹{{INV2_GRAND_TOTAL}}</p></div></div>
<h2>Field-by-Field Comparison (Header/Totals)</h2><table><thead><tr><th>Field</th><th>Invoice 1 Value</th><th>Invoice 2 Value</th><th>Status</th><th>Difference</th></tr></thead><tbody>{{COMPARISON_ROWS}}</tbody></table>
{{ITEM_RECONCILIATION_SECTION}}
{{ITEM_GST_AUTHENTICITY_SECTION}}
<h2>GSTIN Validation</h2><table><thead><tr><th>GSTIN Type</th><th>Inv1 GSTIN (Validation)</th><th>Inv2 GSTIN (Validation)</th><th>Status</th></tr></thead><tbody>{{GSTIN_VALIDATION_ROWS}}</tbody></table>
</div></body></html>"""

    def generate_reconciliation_report(self, comparison_result, inv1_filename="Invoice1", inv2_filename="Invoice2"):
        # ... (This method remains the same as the last fully working one with the UnboundLocalError fix) ...
        html_template = self.get_html_template()
        status_map = {'PERFECT_MATCH':'perfect_match','GOOD_MATCH':'good_match','PARTIAL_MATCH':'partial_match',
                      'POOR_MATCH':'poor_match','PARTIAL_MATCH (CRITICAL FIELD MISMATCH)':'partial_match',
                      'POOR_MATCH (CRITICAL FIELD MISMATCH)':'poor_match','NO_DATA_FOR_COMPARISON':'no_data_for_comparison'}
        overall_status_class = status_map.get(comparison_result['overall_status'], 'poor_match')
        inv1_disp, inv2_disp = comparison_result['invoice1_mapped'], comparison_result['invoice2_mapped']
        comp_rows = ""
        header_fields = ['invoice_number','invoice_date','supplier_name','supplier_gstin','buyer_name','buyer_gstin','subtotal','total_gst','grand_total']
        all_fields = set(list(comparison_result['matches'].keys()) + list(comparison_result['mismatches'].keys()))
        all_fields = {f for f in all_fields if not f.startswith('item_')} 
        sorted_fields = [f for f in header_fields if f in all_fields] + sorted([f for f in all_fields if f not in header_fields])
        
        for field in sorted_fields:
            if field in comparison_result['matches']:
                current_match_data = comparison_result['matches'][field] 
                val = current_match_data.get('value', current_match_data.get('invoice1'))
                is_amt = 'total' in field or 'amount' in field or 'subtotal' in field
                val_float = self._parse_float(val) 
                val_s = f"₹{val_float:.2f}" if is_amt and isinstance(val_float, (float, int)) else str(val)
                diff_s = f"₹{self._parse_float(current_match_data.get('difference',0)):.2f}" if is_amt and 'difference' in current_match_data else "-"
                comp_rows += f'<tr><td>{field.replace("_"," ").title()}</td><td>{val_s}</td><td>{val_s}</td><td><span class="match">MATCH</span></td><td>{diff_s}</td></tr>'
            elif field in comparison_result['mismatches']:
                data = comparison_result['mismatches'][field]
                inv1_v, inv2_v, diff_v = data.get('invoice1','N/A'), data.get('invoice2','N/A'), data.get('difference','N/A')
                is_amt = 'total' in field or 'amount' in field or 'subtotal' in field
                inv1_float, inv2_float = self._parse_float(inv1_v), self._parse_float(inv2_v) 
                inv1_s = f"₹{inv1_float:.2f}" if is_amt and isinstance(inv1_float, (float, int)) else str(inv1_v)
                inv2_s = f"₹{inv2_float:.2f}" if is_amt and isinstance(inv2_float, (float, int)) else str(inv2_v)
                diff_float = self._parse_float(diff_v)
                diff_s = f"₹{diff_float:.2f}" if is_amt and isinstance(diff_float,(int,float)) else str(diff_v)
                comp_rows += f'<tr><td>{field.replace("_"," ").title()}</td><td>{inv1_s}</td><td>{inv2_s}</td><td><span class="mismatch">MISMATCH</span></td><td>{diff_s}</td></tr>'
        
        gst_val_rows = ""
        for field, data in comparison_result['gst_calculations'].items():
            if 'validation' in field:
                type_name, gstin1_d, gstin2_d = field.replace('_validation','').replace('_',' ').title(), data.get('gstin1','N/A'), data.get('gstin2','N/A')
                v1_s, v2_s = "✓ Valid" if data['invoice1_valid'] else "✗ Invalid", "✓ Valid" if data['invoice2_valid'] else "✗ Invalid"
                stat = "MATCH" if data['invoice1_valid']==data['invoice2_valid'] and gstin1_d==gstin2_d and data['invoice1_valid'] else "MISMATCH"
                if not data['invoice1_valid'] and not data['invoice2_valid'] and gstin1_d=="N/A" and gstin2_d=="N/A": stat = "N/A (Both Missing)"
                stat_cls = "match" if stat=="MATCH" else ("mismatch" if stat=="MISMATCH" else "neutral")
                gst_val_rows += f'<tr><td>{type_name}</td><td>{gstin1_d} ({v1_s})</td><td>{gstin2_d} ({v2_s})</td><td><span class="{stat_cls}">{stat}</span></td></tr>'
        
        item_rec_sec = ""
        if 'item_reconciliation' in comparison_result and (comparison_result['item_reconciliation'].get('item_matches') or comparison_result['item_reconciliation'].get('unmatched_items')):
            d, rows = comparison_result['item_reconciliation'], ""
            for m in d.get('item_matches',[]):
                i1,i2,c=m['item1'],m['item2'],m['comparison']
                rows+=f"<tr><td><b>MATCHED (Sim:{m['score']:.2f})</b><br/><i>Inv1:</i>{i1.get('description','N/A')}<br/><i>Inv2:</i>{i2.get('description','N/A')}</td>"
                rows+=f"<td>Qty:{i1.get('quantity','N/A')} vs {i2.get('quantity','N/A')}<br/>Rate:₹{self._parse_float(i1.get('rate','0')):.2f} vs ₹{self._parse_float(i2.get('rate','0')):.2f}<br/>Total:₹{self._parse_float(i1.get('total_amount','0')):.2f} vs ₹{self._parse_float(i2.get('total_amount','0')):.2f}</td>"
                comp_det = "".join([f"<span class='match'>{f.title()}:Match</span><br/>" for f,dt in c.get('matches',{}).items()]) + \
                           "".join([f"<span class='mismatch'>{f.title()}:Mismatch(₹{self._parse_float(dt.get('value1','0')):.2f} vs ₹{self._parse_float(dt.get('value2','0')):.2f})</span><br/>" for f,dt in c.get('mismatches',{}).items()])
                rows+=f"<td>{comp_det if comp_det else 'All fields matched'}</td></tr>"
            for u in d.get('unmatched_items',[]):
                rows+=f"<tr><td><b>UNMATCHED ({u['source']})</b><br/>{u['item'].get('description','N/A')}</td>"
                rows+=f"<td>Qty:{u['item'].get('quantity','N/A')}<br/>Rate:₹{self._parse_float(u['item'].get('rate','0')):.2f}<br/>Total:₹{self._parse_float(u['item'].get('total_amount','0')):.2f}</td><td>-</td></tr>"
            item_rec_sec = f"<h2>Item-Level Reconciliation</h2><p>Items Inv1:{d.get('total_items_inv1',0)}|Items Inv2:{d.get('total_items_inv2',0)}|Matched:{len(d.get('item_matches',[]))}</p><table><thead><tr><th>Item Match Status&Desc</th><th>Key Figures(Inv1 vs Inv2)</th><th>Field Comparison</th></tr></thead><tbody>{rows}</tbody></table><hr/>"
        
        item_gst_auth_sec = ""
        if 'item_gst_authenticity_inv1' in comparison_result and comparison_result['item_gst_authenticity_inv1']:
            rows = "".join([f"""<tr><td>{r.get('invoice_item_description','N/A')}</td><td>{r.get('invoice_rate_detected','N/A')}</td><td>{r.get('db_match_description','N/A')}(Score:{r.get('db_match_score','N/A')})</td><td>IGST:{r.get('expected_igst_db','N/A')}%<br/>CGST:{r.get('expected_cgst_db','N/A')}%<br/>SGST:{r.get('expected_sgst_db','N/A')}%</td><td>Effective DB Rate:{r.get('effective_expected_rate_db','N/A')}%</td><td>DB Exempted:{r.get('is_exempted_db','Unknown')}</td><td class='status-{r.get("status","unverified").lower().replace(" ","-").replace("(","_").replace(")","").replace("%","")}'>{r.get('status','N/A')}</td></tr>""" for r in comparison_result['item_gst_authenticity_inv1']])
            item_gst_auth_sec = f"<h2>Item-wise GST Rate Authenticity Check (Inv1 vs DB)</h2><p><i>Note: 'Invoice Rate Detected' depends on OCR. Often N/A.</i></p><table><thead><tr><th>Item(Inv1)</th><th>Rate on Inv.</th><th>DB Match(Desc&Score)</th><th>Expected Rates(DB)</th><th>Effective DB Rate</th><th>DB Exempted?</th><th>Authenticity Status</th></tr></thead><tbody>{rows}</tbody></table><hr/>"
        
        crit_mismatches_str = ", ".join([f.replace("_"," ").title() for f in comparison_result['summary'].get('critical_mismatches', [])])
        replacements = {'{{COMPARISON_ID}}':comparison_result['comparison_id'], '{{GENERATED_DATE}}':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        '{{OVERALL_STATUS_CLASS}}':overall_status_class, '{{OVERALL_STATUS}}':comparison_result['overall_status'].replace('_',' '),
                        '{{CONFIDENCE_SCORE}}':f"{comparison_result['confidence_score']:.1f}", '{{TOTAL_FIELDS}}':str(comparison_result['summary']['total_fields_compared']),
                        '{{FIELDS_MATCHED}}':str(comparison_result['summary']['fields_matched']), '{{FIELDS_MISMATCHED}}':str(comparison_result['summary']['fields_mismatched']),
                        '{{COMPARISON_TYPE}}':comparison_result['comparison_type'].replace('_',' ').title(), '{{CRITICAL_MISMATCHES}}':bool(crit_mismatches_str),
                        '{{CRITICAL_MISMATCHES_STR}}':crit_mismatches_str, '{{INV1_FILENAME}}':inv1_filename, '{{INV2_FILENAME}}':inv2_filename,
                        '{{INV1_NUMBER}}':str(inv1_disp.get('invoice_number','N/A')), '{{INV1_DATE}}':str(inv1_disp.get('invoice_date','N/A')),
                        '{{INV1_SUPPLIER_NAME}}':str(inv1_disp.get('supplier_name','N/A')), '{{INV1_SUPPLIER_GSTIN}}':str(inv1_disp.get('supplier_gstin','N/A')),
                        '{{INV1_BUYER_NAME}}':str(inv1_disp.get('buyer_name','N/A')), '{{INV1_BUYER_GSTIN}}':str(inv1_disp.get('buyer_gstin','N/A')),
                        '{{INV1_GRAND_TOTAL}}':f"{self._parse_float(inv1_disp.get('grand_total',0)):.2f}",
                        '{{INV2_NUMBER}}':str(inv2_disp.get('invoice_number','N/A')), '{{INV2_DATE}}':str(inv2_disp.get('invoice_date','N/A')),
                        '{{INV2_SUPPLIER_NAME}}':str(inv2_disp.get('supplier_name','N/A')), '{{INV2_SUPPLIER_GSTIN}}':str(inv2_disp.get('supplier_gstin','N/A')),
                        '{{INV2_BUYER_NAME}}':str(inv2_disp.get('buyer_name','N/A')), '{{INV2_BUYER_GSTIN}}':str(inv2_disp.get('buyer_gstin','N/A')),
                        '{{INV2_GRAND_TOTAL}}':f"{self._parse_float(inv2_disp.get('grand_total',0)):.2f}",
                        '{{COMPARISON_ROWS}}':comp_rows, '{{GSTIN_VALIDATION_ROWS}}':gst_val_rows,
                        '{{ITEM_RECONCILIATION_SECTION}}':item_rec_sec, '{{ITEM_GST_AUTHENTICITY_SECTION}}':item_gst_auth_sec}
        html_output = html_template
        for placeholder, value in replacements.items():
            html_output = html_output.replace(str(placeholder), str(value))
        return html_output

# --- Functions from productname_to_gst_model.ipynb (for GST Rate Finder) ---
def parse_hsn_pdf_gst_finder(pdf_content_bytes_unused):
    global HSN_PDF_PATH
    # ... (Keep the robust version of this function from the previous full app.py) ...
    data = []
    try:
        pdf_to_parse_path = HSN_PDF_PATH 
        if not os.path.exists(pdf_to_parse_path):
            print(f"GST Finder: HSN PDF file not found at {pdf_to_parse_path}")
            return pd.DataFrame(columns=['HS_Code_PDF', 'Description_PDF'])
        with open(pdf_to_parse_path, "rb") as f:
            pdf_content_bytes = f.read()
        with pdfplumber.open(io.BytesIO(pdf_content_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        header_idx = -1
                        for r_idx, r_content in enumerate(table):
                            if r_content and len(r_content) > 1 and \
                               any(kw in str(r_content[1]).upper() for kw in ['HS CODE','HSN']) and \
                               ('DESCRIPTION' in str(r_content[2]).upper() if len(r_content)>2 else False):
                                header_idx = r_idx; break
                        data_rows = table[header_idx+1:] if header_idx != -1 else table
                        for row in data_rows:
                            if len(row) >= 3:
                                hs, desc = (str(row[1]).replace('\n',' ').strip() if row[1] else None,
                                            str(row[2]).replace('\n',' ').strip() if row[2] else None)
                                if hs and re.fullmatch(r'\d{2,8}', hs.replace(" ","")):
                                    if desc: data.append({'HS_Code_PDF':hs.replace(" ",""), 'Description_PDF':desc})
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        m = re.search(r'^\s*(\d{2,8})\s+(.+)', line)
                        if m:
                            hs, desc = m.group(1).strip(), m.group(2).strip()
                            if hs and desc and len(desc) > 5: data.append({'HS_Code_PDF':hs, 'Description_PDF':desc})
        if not data: print("GST Finder: Warning: No data from HSN PDF."); return pd.DataFrame(columns=['HS_Code_PDF','Description_PDF'])
        df = pd.DataFrame(data)
        df['HS_Code_PDF'] = df['HS_Code_PDF'].astype(str).str.replace(r'\D+','',regex=True).str.strip().str.lower()
        df.dropna(subset=['HS_Code_PDF','Description_PDF'], inplace=True)
        df = df[df['HS_Code_PDF']!='']
        df.drop_duplicates(subset=['HS_Code_PDF'], keep='first', inplace=True)
        print(f"GST Finder: Extracted {len(df)} unique HSN entries from PDF.")
        return df
    except Exception as e: print(f"GST Finder: Error parsing HSN PDF: {e}"); return pd.DataFrame(columns=['HS_Code_PDF','Description_PDF'])


def aggregate_hsn_descriptions_gst_finder(df_hsn_input):
    # ... (Keep the robust version of this function from the previous full app.py) ...
    if df_hsn_input.empty or not all(c in df_hsn_input.columns for c in ['HS_Code_PDF','Description_PDF']):
        return df_hsn_input.copy()
    print("GST Finder: Aggregating HSN descriptions...")
    df = df_hsn_input.copy()
    df['HS_Code_PDF'] = df['HS_Code_PDF'].astype(str)
    df.sort_values(by='HS_Code_PDF', inplace=True)
    df.reset_index(drop=True, inplace=True)
    agg_desc = {}
    unique_codes = sorted(df['HS_Code_PDF'].unique())
    for parent_hs in unique_codes:
        p_desc_series = df[df['HS_Code_PDF']==parent_hs]['Description_PDF']
        if p_desc_series.empty or pd.isna(p_desc_series.iloc[0]): continue
        curr_descs = [p_desc_series.iloc[0]]
        for child_hs in unique_codes:
            if child_hs.startswith(parent_hs) and len(child_hs)>len(parent_hs):
                c_desc_series = df[df['HS_Code_PDF']==child_hs]['Description_PDF']
                if not c_desc_series.empty and pd.notna(c_desc_series.iloc[0]):
                    curr_descs.append(c_desc_series.iloc[0])
        agg_desc[parent_hs] = ". ".join(list(dict.fromkeys(curr_descs)))
    df['Aggregated_Description_PDF'] = df['HS_Code_PDF'].map(agg_desc)
    df['Aggregated_Description_PDF'].fillna(df['Description_PDF'], inplace=True)
    return df


def parse_gst_csv_gst_finder(csv_content_bytes_unused): # Renamed parameter
    global GST_CSV_PATH
    df_gst = None
    encodings = ['utf-8','latin1','cp1252','iso-8859-1']
    try:
        csv_to_parse_path = GST_CSV_PATH 
        if not os.path.exists(csv_to_parse_path):
            print(f"--- DEBUG GST_CSV: File NOT FOUND at {csv_to_parse_path} ---") # DEBUG
            return pd.DataFrame()
        with open(csv_to_parse_path, "rb") as f:
            csv_content_bytes = f.read()
        print(f"--- DEBUG GST_CSV: Read {len(csv_content_bytes)} bytes from {csv_to_parse_path} ---") # DEBUG

        for enc in encodings:
            try:
                df_temp = pd.read_csv(io.BytesIO(csv_content_bytes), encoding=enc)
                # More robust check for useful columns after read
                if not df_temp.empty and any(col for col in df_temp.columns if any(kw in str(col).lower() for kw in ['gst','hsn','code','description','rate'])): # Added str(col)
                    df_gst = df_temp
                    print(f"--- DEBUG GST_CSV: Loaded with encoding: {enc} ---") # DEBUG
                    print(f"--- DEBUG GST_CSV: Initial shape: {df_gst.shape} ---") # DEBUG
                    print(f"--- DEBUG GST_CSV: Initial columns: {df_gst.columns.tolist()} ---") # DEBUG
                    break
            except Exception as read_err:
                # print(f"DEBUG GST_CSV: Failed read with {enc}: {read_err}") # Optional: can be verbose
                pass
        
        if df_gst is None or df_gst.empty: # Check if df_gst is None OR empty
            print("--- DEBUG GST_CSV: Error: Could not decode or read meaningful data from GST CSV. ---") # DEBUG
            return pd.DataFrame()
        
        # --- Column Mapping Debug ---
        map_kws = {'HS_Code_GST':['hsn','chapter/heading/sub-heading/tariffitem','tariff code','hs code'],
                     'Description_GST':['descriptionofgoods','description of goods','description','goods description'],
                     'CGST_Rate':['cgst(%)','cgst rate','cgst'],'SGST_Rate':['sgst/utgst(%)','sgst rate','sgst','utgst'],
                     'IGST_Rate':['igst(%)','igst rate','igst'],'Compensation_Cess_Raw':['compensationcess','cess']}
        rename_map, df_cols_low = {}, {str(c).lower().strip():c for c in df_gst.columns} # Ensure c is str
        
        print(f"--- DEBUG GST_CSV: Lowercase columns for mapping: {list(df_cols_low.keys())} ---") # DEBUG

        for target_col, source_keywords_list in map_kws.items():
            for src_keyword in source_keywords_list:
                if src_keyword in df_cols_low:
                    original_col_name = df_cols_low[src_keyword]
                    if original_col_name not in rename_map: # Avoid remapping if multiple keywords match same original col
                         rename_map[original_col_name] = target_col
                         print(f"--- DEBUG GST_CSV: Mapping '{original_col_name}' to '{target_col}' (keyword: '{src_keyword}') ---") # DEBUG
                         break # Found mapping for this target_col
        
        if rename_map: 
            df_gst.rename(columns=rename_map, inplace=True)
            print(f"--- DEBUG GST_CSV: Columns AFTER rename: {df_gst.columns.tolist()} ---") # DEBUG
        else:
            print("--- DEBUG GST_CSV: No column renames were applied based on mapping keywords. ---") # DEBUG
        
        # --- Default Columns and Selection Debug ---
        defaults = {'HS_Code_GST':'','Description_GST':'','CGST_Rate':'0','SGST_Rate':'0','IGST_Rate':'0','Compensation_Cess_Raw':'Nil'}
        missing_essential_cols = []
        for col,default_val in defaults.items():
            if col not in df_gst.columns: 
                df_gst[col]=default_val
                print(f"--- DEBUG GST_CSV: Added MISSING essential column '{col}' with default. ---") # DEBUG
                if col in ['HS_Code_GST', 'Description_GST', 'IGST_Rate']: # Mark critical missing cols
                    missing_essential_cols.append(col)
        
        if missing_essential_cols:
            print(f"--- DEBUG GST_CSV: CRITICAL - Essential columns were missing and defaulted: {missing_essential_cols}. Data quality may be poor. ---")

        df_gst_processed = df_gst[list(defaults.keys())].copy() # Ensure we only have and process these columns
        print(f"--- DEBUG GST_CSV: Shape after selecting essential columns: {df_gst_processed.shape} ---") # DEBUG
        
        # --- Data Cleaning and Type Conversion Debug ---
        df_gst_processed['HS_Code_GST'] = df_gst_processed['HS_Code_GST'].astype(str).str.replace(r'\D+','',regex=True).str.strip()
        print(f"--- DEBUG GST_CSV: Sample HS_Code_GST after cleaning: {df_gst_processed['HS_Code_GST'].head().tolist()} ---") #DEBUG
        
        for col in ['CGST_Rate','SGST_Rate','IGST_Rate']:
            # Print some raw values before conversion for these rate columns
            if col in df_gst_processed.columns:
                print(f"--- DEBUG GST_CSV: Raw '{col}' before numeric conversion (sample): {df_gst_processed[col].astype(str).head().tolist()} ---") # DEBUG
            df_gst_processed[col] = pd.to_numeric(df_gst_processed[col].astype(str).str.replace('%','').str.strip(),errors='coerce').fillna(0.0)
            if col in df_gst_processed.columns:
                print(f"--- DEBUG GST_CSV: Converted '{col}' (sample): {df_gst_processed[col].head().tolist()} ---") # DEBUG
        
        # --- Parse Cess Debug ---
        # (Corrected parse_cess function from previous step should be here)
        def parse_cess(val):
            s = str(val).lower().strip()
            if not s or s in ['no', 'false', 'nil', 'exempt', 'exempted', '0', '0%', '0.0', '0.0%', '-', 'na', 'n.a.']:
                return 0.0, False
            m_pct = re.search(r'(\d+\.?\d*)\s*%', s)
            if m_pct:
                try: return float(m_pct.group(1)), True
                except ValueError: pass
            m_rate = re.search(r'(\d+\.?\d*)', s)
            if m_rate:
                try: return float(m_rate.group(1)), True
                except ValueError: pass
            if pd.notna(val) and s not in ['no', 'false', 'nil', 'exempt', 'exempted', '0', '0%', '0.0', '0.0%', '-', 'na', 'n.a.']:
                return 0.0, True
            return 0.0, False
        
        if 'Compensation_Cess_Raw' in df_gst_processed.columns:
             print(f"--- DEBUG GST_CSV: Raw 'Compensation_Cess_Raw' (sample): {df_gst_processed['Compensation_Cess_Raw'].head().tolist()} ---") # DEBUG
             parsed_c = df_gst_processed['Compensation_Cess_Raw'].apply(parse_cess)
             df_gst_processed['Compensation_Cess_Rate'], df_gst_processed['Is_Compensation_Cess'] = parsed_c.apply(lambda x:x[0]), parsed_c.apply(lambda x:x[1])
             df_gst_processed.drop(columns=['Compensation_Cess_Raw'],inplace=True)
             print(f"--- DEBUG GST_CSV: Parsed 'Compensation_Cess_Rate' (sample): {df_gst_processed['Compensation_Cess_Rate'].head().tolist()} ---") # DEBUG
             print(f"--- DEBUG GST_CSV: Parsed 'Is_Compensation_Cess' (sample): {df_gst_processed['Is_Compensation_Cess'].head().tolist()} ---") # DEBUG
        else:
             print("--- DEBUG GST_CSV: 'Compensation_Cess_Raw' column not found for parsing. ---") # DEBUG
             df_gst_processed['Compensation_Cess_Rate'] = 0.0
             df_gst_processed['Is_Compensation_Cess'] = False


        df_gst_processed['Is_Exempted']=((df_gst_processed['CGST_Rate']==0)&(df_gst_processed['SGST_Rate']==0)&(df_gst_processed['IGST_Rate']==0)&((~df_gst_processed['Is_Compensation_Cess'])|(df_gst_processed['Compensation_Cess_Rate']==0)))
        
        # --- Final DataFrame Checks ---
        df_gst_processed.dropna(subset=['HS_Code_GST'],inplace=True) # Drop rows where HSN is truly NaN
        df_gst_processed = df_gst_processed[df_gst_processed['HS_Code_GST']!=''] # Remove rows where HSN became empty string after cleaning
        df_gst_processed.drop_duplicates(subset=['HS_Code_GST'],keep='first',inplace=True)
        
        print(f"--- DEBUG GST_CSV: Final processed shape: {df_gst_processed.shape} ---") # DEBUG
        if df_gst_processed.empty:
            print("--- DEBUG GST_CSV: DataFrame is EMPTY after all processing. ---") # DEBUG
        else:
            print(f"--- DEBUG GST_CSV: Final processed head:\n{df_gst_processed.head().to_string()} ---") # DEBUG
        
        print(f"GST Finder: Processed {len(df_gst_processed)} unique HSN entries from GST CSV.")
        return df_gst_processed
    except Exception as e: 
        print(f"GST Finder: Error parsing GST CSV (outer try-except): {e}")
        import traceback
        traceback.print_exc() 
        return pd.DataFrame()

# --- Model and Data Loading Function (MAIN SETUP) ---
def load_all_resources():
    global donut_processor, donut_model
    global df_merged_gst_finder, sbert_model_gst_finder, corpus_embeddings_gst_finder, qa_pipeline_gst_finder
    global all_descriptions_for_fuzzy_gst_finder
    # TF-IDF globals are removed as we are reverting to SBERT for finder
    # global tfidf_vectorizer_gst_finder, tfidf_matrix_gst_finder 

    print("Loading DONUT (SROIE) model for invoice parsing...")
    try:
        donut_processor = DonutProcessor.from_pretrained(DONUT_MODEL_NAME)
        donut_model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_NAME)
        if torch.cuda.is_available(): donut_model.to("cuda")
        print("DONUT (SROIE) model and processor loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR loading DONUT model: {e}. Invoice reconciliation will be impaired.")
        donut_processor, donut_model = None, None

    print("\nLoading resources for GST Rate Finder (SBERT + QA)...")
    try:
        df_hsn = parse_hsn_pdf_gst_finder(None)
        df_gst = parse_gst_csv_gst_finder(None)

        if df_gst.empty: 
            print("CRITICAL: GST CSV data (for finder) failed. Finder will not work."); 
            df_merged_gst_finder = pd.DataFrame() # Ensure it's empty
            sbert_model_gst_finder = None
            corpus_embeddings_gst_finder = None
            qa_pipeline_gst_finder = None
            all_descriptions_for_fuzzy_gst_finder = []
            return 
        
        if not df_hsn.empty:
            df_hsn_agg = aggregate_hsn_descriptions_gst_finder(df_hsn)
            df_merged_gst_finder = pd.merge(df_gst,df_hsn_agg,left_on='HS_Code_GST',right_on='HS_Code_PDF',how='left')
            desc_g = df_merged_gst_finder.get('Description_GST',pd.Series(dtype='str')).fillna('')
            desc_pa = df_merged_gst_finder.get('Aggregated_Description_PDF',pd.Series(dtype='str')).fillna('')
            df_merged_gst_finder['Combined_Description'] = (desc_g + " " + desc_pa).str.strip()
            df_merged_gst_finder.rename(columns={'HS_Code_GST':'HS_Code'}, inplace=True)
        else:
            df_merged_gst_finder = df_gst.copy()
            df_merged_gst_finder.rename(columns={'HS_Code_GST':'HS_Code'}, inplace=True)
            df_merged_gst_finder['Combined_Description'] = df_merged_gst_finder.get('Description_GST',pd.Series(dtype='str')).fillna('').astype(str).str.strip()

        final_cols = ['HS_Code','Combined_Description','Description_GST','Description_PDF','Aggregated_Description_PDF',
                      'CGST_Rate','SGST_Rate','IGST_Rate','Is_Compensation_Cess','Compensation_Cess_Rate','Is_Exempted']
        # Ensure all final_cols exist, add them with None if not (important for .get() later)
        for col_fc in final_cols:
            if col_fc not in df_merged_gst_finder.columns:
                df_merged_gst_finder[col_fc] = None # Or appropriate default like 0 for rates, False for booleans
        df_merged_gst_finder = df_merged_gst_finder[[c for c in final_cols if c in df_merged_gst_finder.columns]].copy() # Defensive copy
        
        if 'Combined_Description' in df_merged_gst_finder.columns:
            df_merged_gst_finder['Combined_Description'] = df_merged_gst_finder['Combined_Description'].astype(str).str.lower().str.strip()
            all_descriptions_for_fuzzy_gst_finder = df_merged_gst_finder['Combined_Description'].unique().tolist()
            
            # SBERT Setup (reinstated)
            if not df_merged_gst_finder['Combined_Description'].dropna().empty:
                print(f"GST Finder: Merged dataset {len(df_merged_gst_finder)} for SBERT embedding.")
                sbert_model_gst_finder = SentenceTransformer(SENTENCE_MODEL_NAME_GST_FINDER) # Using 'model' as in notebook
                corpus_embeddings_gst_finder = sbert_model_gst_finder.encode( # Using 'corpus_embeddings'
                    df_merged_gst_finder['Combined_Description'].tolist(), convert_to_tensor=True, show_progress_bar=True)
                print("GST Finder: SBERT model & embeddings computed.")
            else: 
                print("GST Finder: No valid descriptions for SBERT embedding.")
                sbert_model_gst_finder = None # Ensure it's None if setup failed
                corpus_embeddings_gst_finder = None
        else: 
            print("GST Finder: 'Combined_Description' missing post-merge. SBERT and Fuzzy search impaired.")
            sbert_model_gst_finder = None
            corpus_embeddings_gst_finder = None
        
        qa_pipeline_gst_finder = hf_pipeline('question-answering', model=QA_MODEL_NAME_GST_FINDER, tokenizer=QA_MODEL_NAME_GST_FINDER) # Using 'qa_pipeline'
        print("GST Rate Finder QA pipeline loaded.")
    except Exception as e:
        print(f"Error loading GST Rate Finder resources: {e}")
        import traceback; traceback.print_exc()
        df_merged_gst_finder=pd.DataFrame(); sbert_model_gst_finder=None; corpus_embeddings_gst_finder=None; qa_pipeline_gst_finder=None

# --- DONUT Inference Function ---
def parse_invoice_with_donut(image_path):
    # ... (This function remains the same as the last fully working version) ...
    if donut_model is None or donut_processor is None:
        return {"error": "OCR Model (DONUT/SROIE) not available."}
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = donut_processor(image, return_tensors="pt").pixel_values
        task_prompt = "<s_sroie-v1>" 
        decoder_input_ids = donut_processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pixel_values, decoder_input_ids = pixel_values.to(device), decoder_input_ids.to(device)
        
        outputs = donut_model.generate(pixel_values, decoder_input_ids=decoder_input_ids,
                                     max_length=donut_model.config.decoder.max_position_embeddings,
                                     early_stopping=True, pad_token_id=donut_processor.tokenizer.pad_token_id,
                                     eos_token_id=donut_processor.tokenizer.eos_token_id, use_cache=True,
                                     num_beams=1, bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
                                     return_dict_in_generate=True)
        sequence = donut_processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
        cleaned_sequence = re.sub(f"^{re.escape(task_prompt)}", "", sequence).strip()
        print(f"DONUT raw seq for {os.path.basename(image_path)}: {cleaned_sequence[:300]}...")
        try:
            parsed_json = donut_processor.token2json(cleaned_sequence)
            print(f"DONUT parsed JSON for {os.path.basename(image_path)} (sample): ", str(parsed_json)[:300] + "...")
            return parsed_json
        except Exception as json_e:
            print(f"Error DONUT token2json: {json_e}. Raw: {cleaned_sequence[:300]}...")
            return {"error": "Failed to parse OCR output to JSON.", "raw_text": cleaned_sequence}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": f"OCR Error: {str(e)}"}

# --- GST Rate Finder Logic (adapted from notebook's get_gst_with_transformer_qa) ---
def get_gst_rates_for_product_page_query(user_query):
    # This function now directly implements the SBERT retrieval + QA logic
    # It uses the global variables: df_merged_gst_finder, sbert_model_gst_finder, 
    # corpus_embeddings_gst_finder, qa_pipeline_gst_finder
    
    if df_merged_gst_finder.empty: return [{"error": "GST data not loaded."}]
    if sbert_model_gst_finder is None or corpus_embeddings_gst_finder is None: return [{"error": "Semantic search model not loaded for GST finder."}]
    if qa_pipeline_gst_finder is None: return [{"error": "QA model not loaded for GST finder."}]
    if not user_query or not user_query.strip(): return [{"error": "Query is empty."}]

    print(f"GST Finder (SBERT Path): Processing query: '{user_query}'")
    try:
        # Mimic notebook's variable names for this section:
        # model = sbert_model_gst_finder
        # corpus_embeddings = corpus_embeddings_gst_finder
        # qa_pipeline = qa_pipeline_gst_finder
        # df_merged = df_merged_gst_finder 
        # util = sbert_util # Already imported as sbert_util

        embedding = sbert_model_gst_finder.encode(user_query.lower().strip(), convert_to_tensor=True)
        if corpus_embeddings_gst_finder.nelement() == 0: return [{"error":"Corpus embeddings empty."}]
        cos_scores = sbert_util.cos_sim(embedding, corpus_embeddings_gst_finder)[0]
        
        # Get top 1 result for QA context (as per notebook's `retrieved[:1]`)
        k_top = min(1, len(cos_scores)) 
        if k_top == 0: return [{"error": "No items in corpus to compare."}]
        top_results = torch.topk(cos_scores, k=k_top)

    except Exception as e:
        print(f"Error during SBERT search for query '{user_query}': {e}")
        return [{"error": f"Error during semantic search: {str(e)}"}]

    sbert_relevance_threshold = 0.35 # Threshold from notebook was 0.3, slightly higher for web
    results_for_template = []

    if not top_results.values.numel() or top_results.values[0].item() < sbert_relevance_threshold:
        results_for_template.append({"error": f"No sufficiently relevant items found for '{user_query}' (Top SBERT score: {top_results.values[0].item():.2f} < {sbert_relevance_threshold}). Try rephrasing."})
        return results_for_template

    # Process the top retrieved item (mimicking notebook's loop `for item in retrieved[:1]:`)
    score_val, idx_val = top_results.values[0].item(), top_results.indices[0].item()
    
    row = df_merged_gst_finder.iloc[idx_val] # Use df_merged_gst_finder
    desc_from_db = row.get('Combined_Description', 'N/A')
    hs_from_db = row.get('HS_Code', 'N/A')
    
    context = (f"The product is '{desc_from_db}' with HS Code {hs_from_db}. "
               f"CGST rate is {row.get('CGST_Rate', 'unknown')} percent. "
               f"SGST or UTGST rate is {row.get('SGST_Rate', 'unknown')} percent. "
               f"IGST rate is {row.get('IGST_Rate', 'unknown')} percent. "
               f"Compensation cess is {'applicable' if row.get('Is_Compensation_Cess') else 'not applicable'}. "
               f"If applicable, compensation cess rate is {row.get('Compensation_Cess_Rate', '0')} percent. "
               f"The product is {'exempted from tax' if row.get('Is_Exempted') else 'not exempted from tax'}.")

    print(f"GST Finder (SBERT Path): Analyzing: {desc_from_db} (HS: {hs_from_db}), SBERT Score: {score_val:.2f}")
    print(f"GST Finder (SBERT Path): Context for QA: {context}")


    questions = {"CGST": "What is the CGST rate?", "SGST": "What is the SGST or UTGST rate?",
                 "IGST": "What is the IGST rate?", "Cess_Applicable": "Is compensation cess applicable?",
                 "Cess_Rate": "What is the compensation cess rate?", "Is_Exempted": "Is the product exempted from tax?"}
    
    qa_details_for_display = {} # Mimics notebook's 'details'
    full_qa_debug_info = f"SBERT Retrieved Context (Score: {score_val:.2f}):\nProduct: {desc_from_db} (HS: {hs_from_db})\nDB Rates -> CGST: {row.get('CGST_Rate')}, SGST: {row.get('SGST_Rate')}, IGST: {row.get('IGST_Rate')}\n\nQA Process:\n"

    for key, q_text in questions.items():
        try:
            res = qa_pipeline_gst_finder({'question': q_text, 'context': context}) # Use qa_pipeline_gst_finder
            answer, qa_score = res['answer'], res['score']
            # Try to clean answer for rates
            if "rate" in q_text.lower():
                num_match = re.search(r'(\d+\.?\d*)', answer)
                if num_match: answer = num_match.group(1)
            
            qa_details_for_display[key] = {'answer': answer, 'score': qa_score}
            full_qa_debug_info += f"Q: {q_text}\nA: {answer} (Conf: {qa_score:.2f})\n"
        except Exception as e:
            print(f"QA Error for '{q_text}': {e}")
            qa_details_for_display[key] = {'answer': 'Error in QA', 'score': 0.0}
            full_qa_debug_info += f"Q: {q_text}\nA: Error in QA\n"
    
    # Format for the webpage, similar to notebook's printout
    cgst_ans = qa_details_for_display.get('CGST',{}).get('answer','N/A')
    sgst_ans = qa_details_for_display.get('SGST',{}).get('answer','N/A')
    igst_ans = qa_details_for_display.get('IGST',{}).get('answer','N/A')
    cess_app_ans = qa_details_for_display.get('Cess_Applicable',{}).get('answer','N/A').lower()
    cess_rate_ans = qa_details_for_display.get('Cess_Rate',{}).get('answer','N/A')
    exempted_ans = qa_details_for_display.get('Is_Exempted',{}).get('answer','N/A').lower()

    results_for_template.append({
        "description_db": desc_from_db.capitalize(), "hs_code_db": hs_from_db,
        "cgst_rate": f"{cgst_ans}%" if cgst_ans not in ['N/A','Error in QA'] else cgst_ans,
        "sgst_rate": f"{sgst_ans}%" if sgst_ans not in ['N/A','Error in QA'] else sgst_ans,
        "igst_rate": f"{igst_ans}%" if igst_ans not in ['N/A','Error in QA'] else igst_ans,
        "cess_applicable": "Yes" if any(kw in cess_app_ans for kw in ['yes','applicable']) else ("No" if "no" in cess_app_ans else "Unknown"),
        "cess_rate": (f"{cess_rate_ans}%" if cess_rate_ans not in ['N/A','Error in QA'] else cess_rate_ans) if any(kw in cess_app_ans for kw in ['yes','applicable']) else "N/A",
        "is_exempted": "Yes" if any(kw in exempted_ans for kw in ['yes','exempted']) else ("No" if "no" in exempted_ans else "Unknown"),
        "retrieval_score": f"{score_val:.2f} (SBERT)",
        "qa_debug_info": full_qa_debug_info.strip()
    })
    
    if not results_for_template: # Should not happen if threshold was passed, but as a fallback
        results_for_template.append({"error": f"Could not determine GST profile for '{user_query}' after SBERT+QA."})
    return results_for_template


# --- Flask Routes ---
@app.route('/')
def home(): return render_template('index.html')

@app.route('/reconcile', methods=['GET', 'POST'])
def reconcile_page():
    # ... (This route remains the same as the last fully working version) ...
    if request.method == 'POST':
        if 'invoice1' not in request.files or 'invoice2' not in request.files:
            flash('Both invoice files are required!', 'error'); return redirect(request.url)
        file1, file2 = request.files['invoice1'], request.files['invoice2']
        if not file1.filename or not file2.filename :
            flash('One or both files not selected!', 'error'); return redirect(request.url)
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename): # allowed_file() is defined
            fname1_orig, fname2_orig = secure_filename(file1.filename), secure_filename(file2.filename)
            uid = uuid.uuid4().hex[:8]
            ext1,ext2 = fname1_orig.rsplit('.',1)[1], fname2_orig.rsplit('.',1)[1]
            fpath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_inv1.{ext1}")
            fpath2 = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_inv2.{ext2}")
            try:
                file1.save(fpath1); file2.save(fpath2)
                flash(f'Files uploaded. Processing {fname1_orig} and {fname2_orig}...', 'info')
                data1_donut, data2_donut = parse_invoice_with_donut(fpath1), parse_invoice_with_donut(fpath2)
                ocr_errs = []
                if "error" in data1_donut: ocr_errs.append(f"Inv1 OCR: {data1_donut['error']} Raw: {data1_donut.get('raw_text','')[:100]}...")
                if "error" in data2_donut: ocr_errs.append(f"Inv2 OCR: {data2_donut['error']} Raw: {data2_donut.get('raw_text','')[:100]}...")
                if ocr_errs:
                    flash(" ".join(ocr_errs), 'error')
                    return render_template('reconcile.html', ocr_error1_detail=data1_donut, ocr_error2_detail=data2_donut)
                
                reconciler = GSTReconciliationEngine()
                comp_res = reconciler.compare_invoices(data1_donut, data2_donut)
                report_html = reconciler.generate_reconciliation_report(comp_res, inv1_filename=fname1_orig, inv2_filename=fname2_orig)
                return render_template('reconciliation_report_display.html', report_content=report_html)
            except Exception as e:
                flash(f'Processing error: {str(e)}', 'error'); import traceback; traceback.print_exc()
                return redirect(request.url)
        else: flash('Allowed file types: png, jpg, jpeg, pdf', 'error'); return redirect(request.url)
    return render_template('reconcile.html')

@app.route('/find-gst-rate', methods=['GET', 'POST'])
def find_gst_rate_page():
    results, query = None, ""
    if request.method == 'POST':
        query = request.form.get('query','').strip()
        if query:
            # Check if SBERT resources are loaded
            if df_merged_gst_finder.empty or sbert_model_gst_finder is None or corpus_embeddings_gst_finder is None or qa_pipeline_gst_finder is None:
                 results = [{"error":"GST Rate Finder service (SBERT/QA) not fully initialized. Check server logs."}]
            else: 
                results = get_gst_rates_for_product_page_query(query) # Calls the SBERT+QA logic
        else: results = [{"error":"Please enter a product description."}]
    return render_template('find_gst_rate.html', results=results, query=query)

@app.route('/uploads/<filename>')
def uploaded_file(filename): return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    print("Starting application, loading resources...")
    load_all_resources()
    
    if donut_model is None: print("\nWARNING: DONUT/SROIE model (OCR) failed. Reconciliation impaired.")
    else: print("\nINFO: DONUT/SROIE model loaded.")
    
    if df_merged_gst_finder.empty: print("\nWARNING: GST Rate Finder data (df_merged_gst_finder) is empty.")
    if sbert_model_gst_finder is None or corpus_embeddings_gst_finder is None:
         print("\nWARNING: SBERT components for GST Rate Finder failed to load.")
    if qa_pipeline_gst_finder is None: print("\nWARNING: QA pipeline for GST Rate Finder failed to load.")
    
    if not df_merged_gst_finder.empty and sbert_model_gst_finder is not None and qa_pipeline_gst_finder is not None:
         print("\nINFO: GST Rate Finder resources (SBERT & QA) appear loaded.")
    else: print("\nERROR: One or more critical resources for GST Rate Finder (SBERT/QA) failed to load.")

    print("\nApp setup complete. Starting Flask server...")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)