# app.py
# Standard library imports
import os
import io
import re
import json
import uuid
from datetime import datetime

# Third-party library imports
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd
import pdfplumber
from fuzzywuzzy import process as fuzzy_process
import torch
import numpy as np

# Sentence Transformers for SBERT (Semantic Search)
from sentence_transformers import SentenceTransformer, util as sbert_util

# Hugging Face Transformers imports (QA model and DONUT)
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    pipeline as hf_pipeline
)

# --- Configuration Constants ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
HSN_PDF_PATH = r"HSN-Codes-for-GST-Enrolment.pdf"
GST_CSV_PATH = r"cbic_gst_goods_rates_exact.csv"

SENTENCE_MODEL_NAME_GST_FINDER = 'all-mpnet-base-v2'
QA_MODEL_NAME_GST_FINDER = 'deepset/roberta-base-squad2'
DONUT_MODEL_NAME = "philschmid/donut-base-sroie"

# --- Global Variables ---
df_merged_gst_finder = pd.DataFrame()
sbert_model_gst_finder = None
corpus_embeddings_gst_finder = None
qa_pipeline_gst_finder = None
all_descriptions_for_fuzzy_gst_finder = []
donut_processor = None
donut_model = None

# --- Flask Application Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_very_secret_key_please_change_this_for_production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Core Logic: GST Reconciliation Engine ---
class GSTReconciliationEngine:
    def __init__(self, tolerance=0.01):
        self.tolerance = tolerance
    def _parse_float(self, value, default=0.0):
        if value is None: return default
        try:
            cleaned_value = str(value).replace('%', ''); cleaned_value = re.sub(r'[^\d\.\-]', '', cleaned_value)
            return float(cleaned_value) if cleaned_value and cleaned_value != '-' else default
        except (ValueError, TypeError): return default
    def validate_gstin(self, gstin):
        if not gstin or len(str(gstin)) != 15: return False
        pattern = r'^[0-9]{2}[A-Z0-9]{10}[0-9][A-Z][0-9A-Z]$'; return bool(re.match(pattern, str(gstin)))
    def map_donut_to_reconciler_format(self, donut_output):
        if not isinstance(donut_output, dict): return {}
        mapped = {}; mapped['invoice_number'] = donut_output.get('id', donut_output.get('doc_id', None))
        mapped['invoice_date'] = donut_output.get('date', None); mapped['supplier_name'] = donut_output.get('company', donut_output.get('store_name', None))
        mapped['supplier_address'] = donut_output.get('address', None); mapped['supplier_gstin'] = donut_output.get('vat_no', donut_output.get('gstin', None))
        mapped['buyer_name'] = None; mapped['buyer_gstin'] = None
        mapped['subtotal'] = self._parse_float(donut_output.get('sub_total', donut_output.get('subtotal', 0)))
        mapped['total_gst'] = self._parse_float(donut_output.get('tax', donut_output.get('vat', 0)))
        mapped['grand_total'] = self._parse_float(donut_output.get('total', 0)); mapped_items = []
        donut_menu_items = donut_output.get('menu', [])
        if isinstance(donut_menu_items, list):
            for item_obj in donut_menu_items:
                if not isinstance(item_obj, dict): continue
                name = item_obj.get('nm', item_obj.get('name', "Unknown Item")); qty_str = str(item_obj.get('cnt', item_obj.get('qty', '1')))
                price_str = str(item_obj.get('price', '0')); quantity = self._parse_float(qty_str, 1.0)
                total_price_for_item_line = self._parse_float(price_str, 0.0)
                unit_price = (total_price_for_item_line / quantity) if quantity > 0 else total_price_for_item_line
                mapped_items.append({'description': str(name).strip() if name else "Unknown Item",'quantity': quantity, 'rate': unit_price, 'amount': total_price_for_item_line,'gst_rate': 0.0, 'gst_amount': 0.0, 'total_amount': total_price_for_item_line})
        elif isinstance(donut_menu_items, dict):
            item_names = donut_menu_items.get('nm', donut_menu_items.get('name', [])); item_prices = donut_menu_items.get('price', [])
            item_quantities = donut_menu_items.get('cnt', donut_menu_items.get('qty', []))
            if isinstance(item_names, list):
                for i, name_val in enumerate(item_names):
                    qty_str = item_quantities[i] if i < len(item_quantities) else '1'; price_str = item_prices[i] if i < len(item_prices) else '0'
                    quantity = self._parse_float(qty_str, 1.0); total_price_for_item_line = self._parse_float(price_str, 0.0)
                    unit_price = (total_price_for_item_line / quantity) if quantity > 0 else total_price_for_item_line
                    mapped_items.append({'description': str(name_val).strip() if name_val else "Unknown Item",'quantity': quantity, 'rate': unit_price, 'amount': total_price_for_item_line,'gst_rate': 0.0, 'gst_amount': 0.0, 'total_amount': total_price_for_item_line})
        mapped['items'] = mapped_items; # print(f"--- MAPPED ITEMS for Reconciler (Count: {len(mapped_items)}) ---")
        # if mapped_items:
        #     for item_idx, item_val in enumerate(mapped_items): print(f"Item {item_idx + 1}: {item_val}")
        # else: print("No items mapped from DONUT.")
        # print(f"--- END MAPPED ITEMS DEBUG ---");
        return mapped
    def _compare_basic_fields(self, inv1, inv2, res):
        for f in ['invoice_number','invoice_date','supplier_name','supplier_gstin','buyer_name','buyer_gstin']:
            v1,v2=(str(inv1.get(f,'')).strip().upper(), str(inv2.get(f,'')).strip().upper())
            if v1==v2 and v1!="": res['matches'][f]={'value':v1,'status':'MATCH'}
            elif v1!=v2: res['mismatches'][f]={'invoice1':v1 or "N/A",'invoice2':v2 or "N/A",'status':'MISMATCH'}
    def _compare_gst_details(self, inv1, inv2, res):
        for f in ['supplier_gstin','buyer_gstin']:
            g1,g2=inv1.get(f),inv2.get(f)
            res['gst_calculations'][f'{f}_validation']={'invoice1_valid':self.validate_gstin(g1) if g1 else False,'invoice2_valid':self.validate_gstin(g2) if g2 else False,'gstin1':g1 or "N/A",'gstin2':g2 or "N/A"}
    def _reconcile_amounts(self, inv1, inv2, res):
        for f in ['subtotal','total_gst','grand_total']:
            a1,a2=self._parse_float(inv1.get(f,0)),self._parse_float(inv2.get(f,0)); diff=abs(a1-a2)
            if a1==0 and a2==0: continue
            if diff<=self.tolerance: res['matches'][f]={'invoice1':a1,'invoice2':a2,'difference':diff,'status':'MATCH'}
            else: res['mismatches'][f]={'invoice1':a1,'invoice2':a2,'difference':diff,'status':'MISMATCH'}
    def _reconcile_line_items(self,items1,items2,res):
        res['item_reconciliation']={'total_items_inv1':len(items1),'total_items_inv2':len(items2),'item_matches':[],'unmatched_items':[]}
        matched_idx2=set()
        for i1,it1 in enumerate(items1):
            d1=str(it1.get('description','')).lower().strip();
            if not d1: continue
            best_idx2,best_score = -1,0.0
            for i2,it2 in enumerate(items2):
                if i2 in matched_idx2: continue
                d2=str(it2.get('description','')).lower().strip();
                if not d2: continue
                score=fuzzy_process.extractOne(d1,[d2])[1]/100.0
                if score > best_score: best_score,best_idx2=score,i2
            if best_idx2!=-1 and best_score >=0.70: matched_idx2.add(best_idx2); comp=self._compare_single_item(it1,items2[best_idx2]); res['item_reconciliation']['item_matches'].append({'item1':it1,'item2':items2[best_idx2],'comparison':comp,'score':best_score})
            else: res['item_reconciliation']['unmatched_items'].append({'source':'invoice1','item':it1})
        for i2,it2 in enumerate(items2):
            if i2 not in matched_idx2: res['item_reconciliation']['unmatched_items'].append({'source':'invoice2','item':it2})
    def _compare_single_item(self,it1,it2):
        comp={'matches':{},'mismatches':{}}
        for f in ['quantity','rate','amount','total_amount']:
            v1,v2=self._parse_float(it1.get(f,0)),self._parse_float(it2.get(f,0))
            if v1==0 and v2==0: continue
            diff=abs(v1-v2); tol=self.tolerance*max(abs(v1),abs(v2)) if 'amount' in f or 'rate' in f else self.tolerance
            if diff<=tol: comp['matches'][f]={'value1':v1,'value2':v2,'difference':diff}
            else: comp['mismatches'][f]={'value1':v1,'value2':v2,'difference':diff}
        return comp
    def _get_expected_gst_profile_for_item(self, description): # SBERT + QA version
        global df_merged_gst_finder, sbert_model_gst_finder, corpus_embeddings_gst_finder, qa_pipeline_gst_finder, all_descriptions_for_fuzzy_gst_finder
        if df_merged_gst_finder.empty: return {"error": "GST Finder: DB not loaded.", "db_description": description}
        if qa_pipeline_gst_finder is None: return {"error": "GST Finder: QA model not loaded.", "db_description": description}
        if not description or not str(description).strip(): return {"error": "GST Finder: Please provide product description.", "db_description": description}
        query = str(description).lower().strip(); retrieved_row_info = None
        if sbert_model_gst_finder is not None and \
           corpus_embeddings_gst_finder is not None and corpus_embeddings_gst_finder.nelement() > 0:
            try:
                # print(f"DEBUG (_get_expected): SBERT model type: {type(sbert_model_gst_finder)}")
                query_embedding = sbert_model_gst_finder.encode(query, convert_to_tensor=True)
                cos_scores = sbert_util.cos_sim(query_embedding, corpus_embeddings_gst_finder)[0]
                if len(cos_scores) > 0:
                    top_result = torch.topk(cos_scores, k=1)
                    sbert_score, sbert_idx = top_result.values[0].item(), top_result.indices[0].item()
                    sbert_threshold = 0.30
                    if sbert_score >= sbert_threshold:
                        retrieved_row_info = {'row': df_merged_gst_finder.iloc[sbert_idx], 'score': sbert_score, 'match_type': 'SBERT'}
            except Exception as e: print(f"ERROR SBERT in _get_expected_gst_profile: {e}")
        if retrieved_row_info is None and all_descriptions_for_fuzzy_gst_finder:
            best_fuzzy = fuzzy_process.extractOne(query, all_descriptions_for_fuzzy_gst_finder)
            if best_fuzzy:
                fuzzy_desc, fuzzy_score_0_100 = best_fuzzy[0], best_fuzzy[1]; fuzzy_score = fuzzy_score_0_100 / 100.0; fuzzy_threshold_internal = 0.75
                if fuzzy_score >= fuzzy_threshold_internal:
                    matched_rows = df_merged_gst_finder[df_merged_gst_finder['Combined_Description'] == fuzzy_desc]
                    if not matched_rows.empty: retrieved_row_info = {'row': matched_rows.iloc[0], 'score': fuzzy_score, 'match_type': 'Fuzzy'}
        if retrieved_row_info is None: return {"error": f"Could not confidently match '{description}' in DB.", "db_description": description, "best_match_score_debug": "N/A"}
        
        row_from_db = retrieved_row_info['row']; match_score = retrieved_row_info['score']; match_type = retrieved_row_info['match_type']
        db_desc_match = row_from_db.get('Combined_Description', 'N/A'); retrieved_hs_code_specific = str(row_from_db.get('HS_Code', 'N/A')).strip()
        retrieved_hs_code_specific_cleaned = re.sub(r'\D', '', retrieved_hs_code_specific).lower()
        row_with_rates = pd.Series(dtype='object'); hs_code_used_for_rates = "N/A (Rates not found)"; potential_hs_for_rates = []
        if len(retrieved_hs_code_specific_cleaned) >= 2:
            potential_hs_for_rates.append(retrieved_hs_code_specific_cleaned)
            if len(retrieved_hs_code_specific_cleaned) > 6: potential_hs_for_rates.append(retrieved_hs_code_specific_cleaned[:6])
            if len(retrieved_hs_code_specific_cleaned) > 4: potential_hs_for_rates.append(retrieved_hs_code_specific_cleaned[:4])
            if len(retrieved_hs_code_specific_cleaned) > 2: potential_hs_for_rates.append(retrieved_hs_code_specific_cleaned[:2])
        potential_hs_for_rates = sorted(list(set(potential_hs_for_rates)), key=len, reverse=True)
        for hs_to_check_for_rates in potential_hs_for_rates:
            if not hs_to_check_for_rates: continue
            temp_df = df_merged_gst_finder[df_merged_gst_finder['HS_Code'].astype(str).str.strip() == hs_to_check_for_rates]
            if not temp_df.empty: row_with_rates = temp_df.iloc[0].copy(); hs_code_used_for_rates = hs_to_check_for_rates; break
        if row_with_rates.empty: default_rate_data = {'CGST_Rate':'unknown','SGST_Rate':'unknown','IGST_Rate':'unknown','Is_Compensation_Cess':False,'Compensation_Cess_Rate':'0','Is_Exempted':False}; row_with_rates = pd.Series(default_rate_data)
        
        context = (f"The product is '{db_desc_match}' with HS Code {retrieved_hs_code_specific}. "
                   f"CGST is {row_with_rates.get('CGST_Rate', 'unknown')}%. "
                   f"SGST/UTGST is {row_with_rates.get('SGST_Rate', 'unknown')}%. "
                   f"IGST is {row_with_rates.get('IGST_Rate', 'unknown')}%. "
                   f"Compensation cess is {'applicable' if row_with_rates.get('Is_Compensation_Cess') else 'not applicable'}. "
                   f"The compensation cess rate is {row_with_rates.get('Compensation_Cess_Rate', '0')}%. "
                   f"The product is {'exempted' if row_with_rates.get('Is_Exempted') else 'not exempted'}.")
        qa_results = {}; questions_map = {"igst_qa": "What is the IGST rate?", "cgst_qa": "What is the CGST rate?", "sgst_qa": "What is the SGST rate?", "is_exempted_qa": "Is the product exempted from tax?"}
        for key, q_text in questions_map.items():
            ans = qa_pipeline_gst_finder(question=q_text, context=context); answer_text, num_match = ans['answer'], re.search(r'(\d+\.?\d*)', ans['answer'])
            qa_results[key] = num_match.group(1) if num_match else answer_text
        return {"db_description": db_desc_match, "hs_code_db": retrieved_hs_code_specific, "score_db": match_score, "match_type_db": match_type,
                "igst_db": self._parse_float(qa_results.get("igst_qa", "0")), "cgst_db": self._parse_float(qa_results.get("cgst_qa", "0")),
                "sgst_db": self._parse_float(qa_results.get("sgst_qa", "0")),
                "is_exempted_db": any(kw in str(qa_results.get("is_exempted_qa", "no")).lower() for kw in ["yes", "exempted"])}
    def verify_item_gst_rates(self,item_inv):
        item_desc=item_inv.get('description',''); exp_prof=self._get_expected_gst_profile_for_item(item_desc)
        stat_dets={"invoice_item_description":item_desc,"invoice_rate_detected":"N/A (SROIE)","db_match_description":exp_prof.get("db_description","N/A"),
                   "db_match_score":f"{exp_prof.get('score_db',0.0):.2f} ({exp_prof.get('match_type_db','N/A')})" if "score_db" in exp_prof else "N/A",
                   "expected_igst_db":exp_prof.get("igst_db","N/A"),"expected_cgst_db":exp_prof.get("cgst_db","N/A"),
                   "expected_sgst_db":exp_prof.get("sgst_db","N/A"),"is_exempted_db":exp_prof.get("is_exempted_db","Unk"),"status":"UNVERIFIED"}
        if "error" in exp_prof: stat_dets["status"]=f"DB_LOOKUP_FAILED ({exp_prof['error']})"; return stat_dets
        db_i,db_c,db_s=exp_prof.get("igst_db",0.0),exp_prof.get("cgst_db",0.0),exp_prof.get("sgst_db",0.0)
        eff_db_rate=db_i if db_i>0 else (db_c+db_s)
        if exp_prof.get("is_exempted_db",False): eff_db_rate=0.0
        stat_dets["effective_expected_rate_db"]=eff_db_rate
        if eff_db_rate>0 and not exp_prof.get("is_exempted_db",False): stat_dets["status"]=f"EXPECTED_RATE_FOUND ({eff_db_rate}%)"
        elif exp_prof.get("is_exempted_db",False): stat_dets["status"]="EXPECTED_EXEMPTED"
        else: stat_dets["status"]="DB_RATE_UNCLEAR_OR_ZERO"
        return stat_dets
    def compare_invoices(self,d1,d2,comp_type="buyer_seller_ocr"):
        res={'comparison_id':f"CMP_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",'comparison_type':comp_type,'invoice1_orig_data':d1,'invoice2_orig_data':d2,
             'invoice1_mapped':self.map_donut_to_reconciler_format(d1),'invoice2_mapped':self.map_donut_to_reconciler_format(d2),
             'matches':{},'mismatches':{},'gst_calculations':{},'item_reconciliation':{},'item_gst_authenticity_inv1':[],
             'overall_status':'UNKNOWN','confidence_score':0.0,'summary':{}}
        self._compare_basic_fields(res['invoice1_mapped'],res['invoice2_mapped'],res); self._compare_gst_details(res['invoice1_mapped'],res['invoice2_mapped'],res)
        self._reconcile_amounts(res['invoice1_mapped'],res['invoice2_mapped'],res)
        itms1,itms2=res['invoice1_mapped'].get('items',[]),res['invoice2_mapped'].get('items',[])
        if itms1 or itms2: self._reconcile_line_items(itms1,itms2,res)
        for it1 in itms1: res['item_gst_authenticity_inv1'].append(self.verify_item_gst_rates(it1))
        self._calculate_overall_status(res); self._generate_summary(res); return res
    def _calculate_overall_status(self,res):
        match_c,mismatch_c=len(res['matches']),len(res['mismatches']); total_basic=match_c+mismatch_c
        if total_basic==0 and not res['invoice1_mapped'].get('items') and not res['invoice2_mapped'].get('items'): res['overall_status'],res['confidence_score']='NO_DATA_FOR_COMPARISON',0.0; return
        base_conf=(match_c/total_basic)*100 if total_basic>0 else 0.0; item_rec_score=0.0
        item_rec_info=res.get('item_reconciliation',{}); tot_itms1,tot_itms2=item_rec_info.get('total_items_inv1',0),item_rec_info.get('total_items_inv2',0)
        num_item_match=len(item_rec_info.get('item_matches',[]))
        if tot_itms1>0 or tot_itms2>0: avg_itms=(tot_itms1+tot_itms2)/2.0 if (tot_itms1+tot_itms2)>0 else 1.0; item_rec_score=(num_item_match/avg_itms)*100 if avg_itms>0 else 0.0; base_conf=(base_conf*0.7)+(item_rec_score*0.3)
        gst_auth_score=0.0; auth_list=res.get('item_gst_authenticity_inv1',[])
        if auth_list:
            succ_look=sum(1 for itm in auth_list if "DB_LOOKUP_FAILED" not in itm.get('status',''))
            if auth_list: gst_auth_score=(succ_look/len(auth_list))*100 if len(auth_list)>0 else 0.0; base_conf=(base_conf*0.8)+(gst_auth_score*0.2)
        res['confidence_score']=round(base_conf,1)
        if len(res['mismatches'])==0 and item_rec_score>=90: res['overall_status']='PERFECT_MATCH'
        elif res['confidence_score']>=75: res['overall_status']='GOOD_MATCH'
        elif res['confidence_score']>=50: res['overall_status']='PARTIAL_MATCH'
        else: res['overall_status']='POOR_MATCH'
        crit_mismatch=any(f in res['mismatches'] and (res['invoice1_mapped'].get(f) or res['invoice2_mapped'].get(f)) for f in ['grand_total','supplier_gstin','buyer_gstin'])
        if crit_mismatch:
            if res['overall_status'] in ['PERFECT_MATCH','GOOD_MATCH']: res['overall_status']='PARTIAL_MATCH (Critical Field Mismatch)'
            elif res['overall_status']=='PARTIAL_MATCH': res['overall_status']='POOR_MATCH (Critical Field Mismatch)'
    def _generate_summary(self,res):
        res['summary']={'total_fields_compared':len(res['matches'])+len(res['mismatches']),'fields_matched':len(res['matches']),'fields_mismatched':len(res['mismatches']),'amount_differences':[],'critical_mismatches':[]}
        for f,d in res['mismatches'].items():
            if 'difference' in d and isinstance(d['difference'],(int,float)): res['summary']['amount_differences'].append({'field':f,'difference':d['difference']})
        for f in ['grand_total','supplier_gstin','buyer_gstin']:
            if f in res['mismatches'] and (res['invoice1_mapped'].get(f) or res['invoice2_mapped'].get(f)): res['summary']['critical_mismatches'].append(f)
    def get_html_template(self): return """<!DOCTYPE html><html><head><title>Reconciliation Report</title><meta name="viewport" content="width=device-width, initial-scale=1.0"><style>body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:0;padding:0;background-color:#f4f7f6;color:#333;}.report-container{max-width:1200px;margin:20px auto;background:white;padding:25px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08);}h1,h2,h3{color:#333;}h1{text-align:center;color:#1a535c;margin-bottom:5px;font-size:1.8em;}.report-meta{text-align:center;font-size:0.9em;color:#555;margin-bottom:20px;}table{width:100%;border-collapse:collapse;margin-bottom:25px;font-size:0.9em;}th,td{border:1px solid #e0e0e0;padding:10px 12px;text-align:left;vertical-align:top;}th{background-color:#f0f4f8;color:#334e68;font-weight:600;}tr:nth-child(even){background-color:#fbfcfc;}.match{color:#27ae60;font-weight:bold;}.mismatch{color:#e74c3c;font-weight:bold;}.neutral{color:#7f8c8d;}.status-badge{display:inline-block;padding:8px 15px;border-radius:15px;color:white;font-weight:bold;font-size:1.1em;margin-bottom:15px;}.status-perfect_match{background-color:#27ae60;}.status-good_match{background-color:#5cb85c;}.status-partial_match{background-color:#f39c12;}.status-poor_match{background-color:#e74c3c;}.status-partial_match-critical-field-mismatch{background-color:#e67e22;}.status-poor_match-critical-field-mismatch{background-color:#c0392b;}.status-no_data_for_comparison{background-color:#7f8c8d;}.invoice-details-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}.invoice-card{background:#f9f9f9;padding:15px;border-radius:5px;border:1px solid #e8e8e8;}.invoice-card h3{margin-top:0;color:#34495e;font-size:1.1em;border-bottom:1px solid #ddd;padding-bottom:5px;margin-bottom:10px;}.summary-card{background-color:#e9f5ff;border-left:4px solid #3498db;padding:12px;margin-bottom:8px;border-radius:4px;font-size:0.95em;}.status-expected_rate_found{color:#16a085;}.status-expected_exempted{color:#27ae60;}.status-db_rate_unclear_or_zero{color:#7f8c8d;}.status-db_lookup_failed{color:#f39c12;font-weight:bold;}.status-unverified{color:#95a5a6;}@media (max-width:768px){.invoice-details-grid{grid-template-columns:1fr;}th,td{padding:8px;}}</style></head><body><div class="report-container"><h1>GST Reconciliation Report</h1><p class="report-meta">Comparison ID: <strong>{{COMPARISON_ID}}</strong> | Generated: {{GENERATED_DATE}}</p><div style="text-align:center;"><span class="status-badge status-{{OVERALL_STATUS_CLASS}}">Overall Status: {{OVERALL_STATUS}} (Confidence: {{CONFIDENCE_SCORE}}%)</span></div><h2>Summary</h2><div class="summary-card">Fields Compared (Header/Totals): {{TOTAL_FIELDS}}, Matched: {{FIELDS_MATCHED}}, Mismatched: {{FIELDS_MISMATCHED}}</div><div class="summary-card">Comparison Type: {{COMPARISON_TYPE}}</div>{% if CRITICAL_MISMATCHES_STR %}<div class="summary-card" style="background-color:#ffebee;border-left-color:#e74c3c;">Critical Mismatches found in: {{CRITICAL_MISMATCHES_STR}}</div>{% else %}<div class="summary-card" style="background-color:#e8f5e9;border-left-color:#27ae60;">No critical header/total mismatches.</div>{% endif %}<h2>Invoice Details</h2><div class="invoice-details-grid"><div class="invoice-card"><h3>Invoice 1 ({{INV1_FILENAME}})</h3><p><strong>Inv No:</strong> {{INV1_NUMBER}} | <strong>Date:</strong> {{INV1_DATE}}</p><p><strong>Supplier:</strong> {{INV1_SUPPLIER_NAME}} (GSTIN: {{INV1_SUPPLIER_GSTIN}})</p><p><strong>Buyer:</strong> {{INV1_BUYER_NAME}} (GSTIN: {{INV1_BUYER_GSTIN}})</p><p><strong>Grand Total:</strong> ₹{{INV1_GRAND_TOTAL}}</p></div><div class="invoice-card"><h3>Invoice 2 ({{INV2_FILENAME}})</h3><p><strong>Inv No:</strong> {{INV2_NUMBER}} | <strong>Date:</strong> {{INV2_DATE}}</p><p><strong>Supplier:</strong> {{INV2_SUPPLIER_NAME}} (GSTIN: {{INV2_SUPPLIER_GSTIN}})</p><p><strong>Buyer:</strong> {{INV2_BUYER_NAME}} (GSTIN: {{INV2_BUYER_GSTIN}})</p><p><strong>Grand Total:</strong> ₹{{INV2_GRAND_TOTAL}}</p></div></div><h2>Field-by-Field Comparison (Header/Totals)</h2><table><thead><tr><th>Field</th><th>Invoice 1 Value</th><th>Invoice 2 Value</th><th>Status</th><th>Difference</th></tr></thead><tbody>{{COMPARISON_ROWS}}</tbody></table>{{ITEM_RECONCILIATION_SECTION}}{{ITEM_GST_AUTHENTICITY_SECTION}}<h2>GSTIN Validation</h2><table><thead><tr><th>GSTIN Type</th><th>Inv1 GSTIN (Validation)</th><th>Inv2 GSTIN (Validation)</th><th>Status</th></tr></thead><tbody>{{GSTIN_VALIDATION_ROWS}}</tbody></table></div></body></html>"""
    def generate_reconciliation_report(self, comparison_result, inv1_filename="Invoice1", inv2_filename="Invoice2"):
        html_template = self.get_html_template(); status_to_class = {'PERFECT_MATCH':'perfect_match','GOOD_MATCH':'good_match','PARTIAL_MATCH':'partial_match','POOR_MATCH':'poor_match','PARTIAL_MATCH (Critical Field Mismatch)':'partial_match-critical-field-mismatch','POOR_MATCH (Critical Field Mismatch)':'poor_match-critical-field-mismatch','NO_DATA_FOR_COMPARISON':'no_data_for_comparison'}
        overall_status_class = status_to_class.get(comparison_result['overall_status'], 'poor_match'); inv1_disp = comparison_result['invoice1_mapped']; inv2_disp = comparison_result['invoice2_mapped']
        comp_rows_html = ""; header_fields_order = ['invoice_number','invoice_date','supplier_name','supplier_gstin','buyer_name','buyer_gstin','subtotal','total_gst','grand_total']; processed_fields = set()
        for field in header_fields_order:
            if field in comparison_result['matches']: data = comparison_result['matches'][field]; val = data.get('value', data.get('invoice1')); is_amount_field = any(kw in field.lower() for kw in ['total', 'amount', 'subtotal']); val_str = f"₹{self._parse_float(val):.2f}" if is_amount_field and isinstance(self._parse_float(val), (float, int)) else str(val); diff_str = f"₹{self._parse_float(data.get('difference',0)):.2f}" if is_amount_field and 'difference' in data else "-"; comp_rows_html += f'<tr><td>{field.replace("_"," ").title()}</td><td>{val_str}</td><td>{val_str}</td><td><span class="match">MATCH</span></td><td>{diff_str}</td></tr>'; processed_fields.add(field)
            elif field in comparison_result['mismatches']: data = comparison_result['mismatches'][field]; v1, v2, diff = data.get('invoice1','N/A'), data.get('invoice2','N/A'), data.get('difference','N/A'); is_amount_field = any(kw in field.lower() for kw in ['total', 'amount', 'subtotal']); v1_str = f"₹{self._parse_float(v1):.2f}" if is_amount_field and isinstance(self._parse_float(v1), (float, int)) else str(v1); v2_str = f"₹{self._parse_float(v2):.2f}" if is_amount_field and isinstance(self._parse_float(v2), (float, int)) else str(v2); diff_str = f"₹{self._parse_float(diff):.2f}" if is_amount_field and isinstance(self._parse_float(diff), (float, int)) else str(diff); comp_rows_html += f'<tr><td>{field.replace("_"," ").title()}</td><td>{v1_str}</td><td>{v2_str}</td><td><span class="mismatch">MISMATCH</span></td><td>{diff_str}</td></tr>'; processed_fields.add(field)
        gst_val_rows_html = "";
        for field_key, data in comparison_result['gst_calculations'].items():
            if 'validation' in field_key: type_name = field_key.replace('_validation','').replace('_',' ').title(); gstin1_val, gstin2_val = data.get('gstin1','N/A'), data.get('gstin2','N/A'); valid1_str = "✓ Valid" if data['invoice1_valid'] else "✗ Invalid"; valid2_str = "✓ Valid" if data['invoice2_valid'] else "✗ Invalid"; status_str, status_cls = "N/A (Both Missing)", "neutral";
            if gstin1_val != "N/A" or gstin2_val != "N/A":
                if data['invoice1_valid'] == data['invoice2_valid'] and gstin1_val == gstin2_val and data['invoice1_valid']: status_str, status_cls = "MATCH", "match"
                else: status_str, status_cls = "MISMATCH", "mismatch"
            gst_val_rows_html += f'<tr><td>{type_name}</td><td>{gstin1_val} ({valid1_str})</td><td>{gstin2_val} ({valid2_str})</td><td><span class="{status_cls}">{status_str}</span></td></tr>'
        item_rec_section_html = ""; item_rec_data = comparison_result.get('item_reconciliation', {})
        if item_rec_data and (item_rec_data.get('item_matches') or item_rec_data.get('unmatched_items')):
            rows = "";
            for match_info in item_rec_data.get('item_matches', []): i1, i2, comp = match_info['item1'], match_info['item2'], match_info['comparison']; rows += f"<tr><td><b>MATCHED (Sim:{match_info['score']:.2f})</b><br/><i>Inv1:</i> {i1.get('description','N/A')}<br/><i>Inv2:</i> {i2.get('description','N/A')}</td>"; rows += f"<td>Qty: {i1.get('quantity','N/A')} vs {i2.get('quantity','N/A')}<br/>Rate: ₹{self._parse_float(i1.get('rate','0')):.2f} vs ₹{self._parse_float(i2.get('rate','0')):.2f}<br/>Line Total: ₹{self._parse_float(i1.get('total_amount','0')):.2f} vs ₹{self._parse_float(i2.get('total_amount','0')):.2f}</td>"; comp_details_html = "".join([f"<span class='match'>{f_name.title()}: Match</span><br/>" for f_name in comp.get('matches',{})]) + "".join([f"<span class='mismatch'>{f_name.title()}: Mismatch (₹{self._parse_float(f_data.get('value1','0')):.2f} vs ₹{self._parse_float(f_data.get('value2','0')):.2f})</span><br/>" for f_name, f_data in comp.get('mismatches',{}).items()]); rows += f"<td>{comp_details_html if comp_details_html else 'All fields matched'}</td></tr>"
            for unmatch_info in item_rec_data.get('unmatched_items', []): item = unmatch_info['item']; rows += f"<tr><td><b>UNMATCHED ({unmatch_info['source']})</b><br/>{item.get('description','N/A')}</td>"; rows += f"<td>Qty: {item.get('quantity','N/A')}<br/>Rate: ₹{self._parse_float(item.get('rate','0')):.2f}<br/>Line Total: ₹{self._parse_float(item.get('total_amount','0')):.2f}</td><td>-</td></tr>"
            item_rec_section_html = f"<h2>Item-Level Reconciliation</h2><p>Items Inv1: {item_rec_data.get('total_items_inv1',0)} | Items Inv2: {item_rec_data.get('total_items_inv2',0)} | Matched Pairs: {len(item_rec_data.get('item_matches',[]))}</p><table><thead><tr><th>Item Match Status & Desc</th><th>Key Figures (Inv1 vs Inv2)</th><th>Field Comparison</th></tr></thead><tbody>{rows}</tbody></table><hr/>"
        item_gst_auth_section_html = ""; auth_results = comparison_result.get('item_gst_authenticity_inv1', [])
        if auth_results:
            rows = "";
            for r_data in auth_results: status_class = str(r_data.get("status","unverified")).lower().replace(" ","-").replace("(","_").replace(")","").replace("%",""); rows += f"""<tr><td>{r_data.get('invoice_item_description','N/A')}</td><td>{r_data.get('invoice_rate_detected','N/A')}</td><td>{r_data.get('db_match_description','N/A')} (Score: {r_data.get('db_match_score','N/A')})</td><td>IGST: {r_data.get('expected_igst_db','N/A')}%<br/>CGST: {r_data.get('expected_cgst_db','N/A')}%<br/>SGST: {r_data.get('expected_sgst_db','N/A')}%</td><td>{r_data.get('effective_expected_rate_db','N/A')}%</td><td>{r_data.get('is_exempted_db','Unknown')}</td><td class='status-{status_class}'>{r_data.get('status','N/A')}</td></tr>"""
            item_gst_auth_section_html = f"<h2>Item-wise GST Rate Authenticity Check (Inv1 vs DB)</h2><p><i>Note: 'Invoice Rate Detected' N/A for SROIE. Status reflects DB lookup.</i></p><table><thead><tr><th>Item(Inv1)</th><th>Rate on Inv.</th><th>DB Match</th><th>Expected Rates(DB)</th><th>Effective DB Rate</th><th>DB Exempted?</th><th>Authenticity Status</th></tr></thead><tbody>{rows}</tbody></table><hr/>"
        crit_mismatches_str_display = ", ".join([f.replace("_"," ").title() for f in comparison_result['summary'].get('critical_mismatches', [])])
        replacements = {'{{COMPARISON_ID}}': comparison_result['comparison_id'],'{{GENERATED_DATE}}': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'{{OVERALL_STATUS_CLASS}}': overall_status_class,'{{OVERALL_STATUS}}': comparison_result['overall_status'].replace('_',' ').title(),'{{CONFIDENCE_SCORE}}': f"{comparison_result['confidence_score']:.1f}",'{{TOTAL_FIELDS}}': str(comparison_result['summary']['total_fields_compared']),'{{FIELDS_MATCHED}}': str(comparison_result['summary']['fields_matched']),'{{FIELDS_MISMATCHED}}': str(comparison_result['summary']['fields_mismatched']),'{{COMPARISON_TYPE}}': comparison_result['comparison_type'].replace('_',' ').title(),'{{CRITICAL_MISMATCHES_STR}}': crit_mismatches_str_display,'{{INV1_FILENAME}}': inv1_filename,'{{INV2_FILENAME}}': inv2_filename,'{{INV1_NUMBER}}': str(inv1_disp.get('invoice_number','N/A')),'{{INV1_DATE}}': str(inv1_disp.get('invoice_date','N/A')),'{{INV1_SUPPLIER_NAME}}': str(inv1_disp.get('supplier_name','N/A')),'{{INV1_SUPPLIER_GSTIN}}': str(inv1_disp.get('supplier_gstin','N/A')),'{{INV1_BUYER_NAME}}': str(inv1_disp.get('buyer_name','N/A')),'{{INV1_BUYER_GSTIN}}': str(inv1_disp.get('buyer_gstin','N/A')),'{{INV1_GRAND_TOTAL}}': f"{self._parse_float(inv1_disp.get('grand_total',0)):.2f}",'{{INV2_NUMBER}}': str(inv2_disp.get('invoice_number','N/A')),'{{INV2_DATE}}': str(inv2_disp.get('invoice_date','N/A')),'{{INV2_SUPPLIER_NAME}}': str(inv2_disp.get('supplier_name','N/A')),'{{INV2_SUPPLIER_GSTIN}}': str(inv2_disp.get('supplier_gstin','N/A')),'{{INV2_BUYER_NAME}}': str(inv2_disp.get('buyer_name','N/A')),'{{INV2_BUYER_GSTIN}}': str(inv2_disp.get('buyer_gstin','N/A')),'{{INV2_GRAND_TOTAL}}': f"{self._parse_float(inv2_disp.get('grand_total',0)):.2f}",'{{COMPARISON_ROWS}}': comp_rows_html,'{{GSTIN_VALIDATION_ROWS}}': gst_val_rows_html,'{{ITEM_RECONCILIATION_SECTION}}': item_rec_section_html,'{{ITEM_GST_AUTHENTICITY_SECTION}}': item_gst_auth_section_html}
        final_html = html_template;
        for p,v in replacements.items(): final_html = final_html.replace(str(p),str(v))
        return final_html

# --- Data Parsing Functions (for GST Rate Finder) ---
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
                        for r_idx, r_content in enumerate(table): # Find header row based on keywords
                            if r_content and len(r_content) > 2: # Need at least 3 columns for SL, HS, DESC
                                col1_text = str(r_content[1] if r_content[1] else "").upper() # Potential HS_CODE Column
                                col2_text = str(r_content[2] if r_content[2] else "").upper() # Potential DESCRIPTION Column
                                if any(kw in col1_text for kw in ['HS CODE', 'HSN']) and 'DESCRIPTION' in col2_text:
                                    header_row_index = r_idx; break
                        data_rows_to_parse = table[header_row_index+1:] if header_row_index != -1 else table
                        for row_idx, row in enumerate(data_rows_to_parse):
                            if len(row) >= 3 :
                                hs_code_raw = str(row[1]).replace('\n',' ').strip() if row[1] else None
                                description_raw = str(row[2]).replace('\n',' ').strip() if row[2] else None
                                if hs_code_raw and description_raw:
                                    cleaned_hs = re.sub(r'[^0-9]', '', hs_code_raw) # Keep only digits
                                    if 2 <= len(cleaned_hs) <= 8:
                                        data.append({'HS_Code_PDF': cleaned_hs, 'Description_PDF': description_raw})
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        match = re.search(r'^\s*(\d[\d\.\s]*\d)\s+([A-Za-z].*)', line) # HS can have dots/spaces, Desc must start with letter
                        if match:
                            hs_code_raw = match.group(1).strip(); description_raw = match.group(2).strip()
                            cleaned_hs = re.sub(r'[^0-9]','', hs_code_raw)
                            if cleaned_hs and description_raw and len(description_raw) > 3 and (2 <= len(cleaned_hs) <= 8) :
                                data.append({'HS_Code_PDF': cleaned_hs, 'Description_PDF': description_raw})
        if not data: print("GST Finder Warning: No data extracted from HSN PDF."); return pd.DataFrame(columns=['HS_Code_PDF','Description_PDF'])
        df_hsn = pd.DataFrame(data)
        df_hsn['HS_Code_PDF'] = df_hsn['HS_Code_PDF'].astype(str).str.lower()
        df_hsn.dropna(subset=['HS_Code_PDF', 'Description_PDF'], inplace=True)
        df_hsn = df_hsn[df_hsn['HS_Code_PDF'] != '']; df_hsn.drop_duplicates(subset=['HS_Code_PDF'], keep='first', inplace=True)
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
    df_hsn['Aggregated_Description_PDF'] = df_hsn['Aggregated_Description_PDF'].fillna(value=df_hsn['Description_PDF'])
    print("GST Finder: HSN description aggregation complete."); return df_hsn

def parse_gst_csv_gst_finder(csv_content_bytes_unused):
    global GST_CSV_PATH; df_gst_raw = None; encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1', 'windows-1252']
    try:
        if not os.path.exists(GST_CSV_PATH): print(f"--- GST_CSV ERROR: File NOT FOUND at '{GST_CSV_PATH}' ---"); return pd.DataFrame()
        with open(GST_CSV_PATH, "rb") as f_in: csv_bytes_content = f_in.read()
        print(f"--- GST_CSV INFO: Read {len(csv_bytes_content)} bytes from '{GST_CSV_PATH}' ---"); successful_encoding = None
        for enc in encodings_to_try:
            print(f"--- GST_CSV INFO: Attempting to read CSV with encoding: '{enc}' ---")
            try:
                df_temp = pd.read_csv(io.BytesIO(csv_bytes_content), encoding=enc)
                if not df_temp.empty and any(any(kw in str(c).lower() for kw in ['gst','hsn','code','desc','tariff','rate']) for c in df_temp.columns):
                    df_gst_raw, successful_encoding = df_temp, enc
                    print(f"--- GST_CSV INFO: Successfully loaded with encoding: '{successful_encoding}' ---"); print(f"--- GST_CSV INFO: Initial shape: {df_gst_raw.shape}, Cols: {df_gst_raw.columns.tolist()} ---"); break
            except UnicodeDecodeError: print(f"--- GST_CSV WARNING: UnicodeDecodeError with encoding '{enc}'. ---")
            except Exception as e: print(f"--- GST_CSV WARNING: Error with '{enc}': {e}. ---")
        if df_gst_raw is None or df_gst_raw.empty:
            print("--- GST_CSV CRITICAL: Could not read. Trying with errors='replace'. ---")
            try:
                df_temp = pd.read_csv(io.BytesIO(csv_bytes_content), encoding='utf-8', errors='replace')
                if not df_temp.empty and any(any(kw in str(c).lower() for kw in ['gst','hsn','code','desc','rate']) for c in df_temp.columns):
                    df_gst_raw = df_temp; print(f"--- GST_CSV WARNING: Loaded with 'utf-8' errors='replace'. Shape: {df_gst_raw.shape} ---")
                else: print("--- GST_CSV CRITICAL: Last resort read also failed. ---"); return pd.DataFrame()
            except Exception as e: print(f"--- GST_CSV CRITICAL: Exception during last resort: {e} ---"); return pd.DataFrame()

        original_hsn_col_name = None; df_cols_map = {str(c).lower().strip(): c for c in df_gst_raw.columns}
        hsn_kws = ['chapter/heading/sub-heading/tariffitem', 'hsn', 'tariff code', 'hs code', 'tariffitem']
        for kw in hsn_kws:
            if kw in df_cols_map: original_hsn_col_name = df_cols_map[kw]; print(f"--- GST_CSV DEBUG: ID'd HSN col: '{original_hsn_col_name}' (via '{kw}') ---"); break
        
        if not original_hsn_col_name: print("--- GST_CSV CRIT ERROR: Could not ID original HSN col. ---"); df_gst_raw['HS_Code_GST_Cleaned'] = ''
        else: print(f"--- GST_CSV DEBUG: Cleaning HSN from col: '{original_hsn_col_name}' ---"); df_gst_raw['HS_Code_GST_Cleaned'] = df_gst_raw[original_hsn_col_name].apply(extract_and_clean_hsn_from_cell); print(f"--- DEBUG GST_CSV: Sample HS_Code_GST_Cleaned: {df_gst_raw['HS_Code_GST_Cleaned'].head().tolist()} ---")

        col_map_kws={'Description_GST':['descriptionofgoods','desc of goods','description','goods desc'],'CGST_Rate':['cgst(%)','cgst rate','cgst'],'SGST_Rate':['sgst/utgst(%)','sgst rate','sgst','utgst'],'IGST_Rate':['igst(%)','igst rate','igst'],'Compensation_Cess_Raw':['compensationcess','cess','comp cess']}
        rename_map = {}
        for target, sources in col_map_kws.items():
            for src_kw in sources:
                if src_kw in df_cols_map:
                    orig_col = df_cols_map[src_kw]
                    if orig_col not in rename_map and orig_col != original_hsn_col_name: rename_map[orig_col] = target; print(f"--- GST_CSV DEBUG: Mapping '{orig_col}' to '{target}' (kw: '{src_kw}') ---"); break
        if rename_map: df_gst_raw.rename(columns=rename_map, inplace=True); print(f"--- GST_CSV DEBUG: Cols AFTER rename (excl HSN): {df_gst_raw.columns.tolist()} ---")
        
        if 'HS_Code_GST_Cleaned' in df_gst_raw.columns: df_gst_raw.rename(columns={'HS_Code_GST_Cleaned': 'HS_Code_GST'}, inplace=True); print(f"--- GST_CSV DEBUG: Renamed 'HS_Code_GST_Cleaned' to 'HS_Code_GST'. ---")
        elif 'HS_Code_GST' not in df_gst_raw.columns: df_gst_raw['HS_Code_GST'] = ''; print(f"--- GST_CSV WARN: Defaulted 'HS_Code_GST' to empty. ---")

        ess_cols={'HS_Code_GST':'','Description_GST':'','CGST_Rate':'0','SGST_Rate':'0','IGST_Rate':'0','Compensation_Cess_Raw':'Nil'}; missing_crit=[]
        for col,default in ess_cols.items():
            if col not in df_gst_raw.columns: df_gst_raw[col]=default; print(f"--- GST_CSV DEBUG: Added MISSING col '{col}'. ---");
            if col in ['Description_GST','IGST_Rate'] and col not in df_gst_raw.columns: missing_crit.append(col) # HS_Code_GST handled
        if missing_crit: print(f"--- GST_CSV CRIT WARN: Other essential cols missing & defaulted: {missing_crit}. ---")
        
        df_proc = df_gst_raw.copy();
        cols_to_select = [col for col in ess_cols.keys() if col in df_proc.columns] 
        df_proc = df_proc[cols_to_select].copy()
        print(f"--- GST_CSV DEBUG: Shape after selecting ess cols (df_proc): {df_proc.shape} ---")
        if 'HS_Code_GST' in df_proc.columns: print(f"--- DEBUG GST_CSV: Sample HS_Code_GST in df_proc: {df_proc['HS_Code_GST'].head().tolist()} ---")
        else: print(f"--- GST_CSV CRIT ERROR: 'HS_Code_GST' not in df_proc! ---"); return pd.DataFrame()

        for col in ['CGST_Rate','SGST_Rate','IGST_Rate']:
            if col in df_proc.columns: df_proc[col] = pd.to_numeric(df_proc[col].astype(str).str.replace('%','').str.strip(),errors='coerce').fillna(0.0)
        
        def parse_cess_value(value): # Corrected definition
            str_val = str(value).lower().strip()
            if not str_val or str_val in ['no', 'false', 'nil', 'exempt', 'exempted', '0', '0%', '0.0', '0.0%', '-', 'na', 'n.a.', '']:
                return 0.0, False
            match_percent = re.search(r'(\d+\.?\d*)\s*%', str_val)
            if match_percent:
                try: return float(match_percent.group(1)), True
                except ValueError: pass
            match_numeric = re.search(r'(\d+\.?\d*)', str_val)
            if match_numeric:
                try: return float(match_numeric.group(1)), True
                except ValueError: pass
            if pd.notna(value): return 0.0, True
            return 0.0, False
            
        if 'Compensation_Cess_Raw' in df_proc.columns:
            # print(f"--- GST_CSV DEBUG: Raw 'Compensation_Cess_Raw' (sample): {df_proc['Compensation_Cess_Raw'].head().tolist()} ---")
            parsed_c = df_proc['Compensation_Cess_Raw'].apply(parse_cess_value) # Uses corrected function
            df_proc['Compensation_Cess_Rate'] = parsed_c.apply(lambda x: x[0])
            df_proc['Is_Compensation_Cess'] = parsed_c.apply(lambda x: x[1])
            df_proc.drop(columns=['Compensation_Cess_Raw'],inplace=True)
            # print(f"--- GST_CSV DEBUG: Parsed 'Compensation_Cess_Rate' (sample): {df_proc['Compensation_Cess_Rate'].head().tolist()} ---")
            # print(f"--- GST_CSV DEBUG: Parsed 'Is_Compensation_Cess' (sample): {df_proc['Is_Compensation_Cess'].head().tolist()} ---")
        else:
             df_proc['Compensation_Cess_Rate'] = 0.0
             df_proc['Is_Compensation_Cess'] = False
             print("--- GST_CSV DEBUG: 'Compensation_Cess_Raw' column not found in df_proc, defaulting Cess. ---")

        df_proc['Is_Exempted']=((df_proc['CGST_Rate']==0)&(df_proc['SGST_Rate']==0)&(df_proc['IGST_Rate']==0)&((~df_proc['Is_Compensation_Cess'])|(df_proc['Compensation_Cess_Rate']==0)))
        
        if 'HS_Code_GST' in df_proc.columns:
            df_proc.dropna(subset=['HS_Code_GST'], inplace=True)
            df_proc = df_proc[df_proc['HS_Code_GST'] != '']
            df_proc.drop_duplicates(subset=['HS_Code_GST'], keep='first', inplace=True)
        else:
            print("--- GST_CSV CRITICAL ERROR: 'HS_Code_GST' missing for final cleanup. Returning empty DataFrame. ---")
            return pd.DataFrame()

        print(f"--- GST_CSV DEBUG: Final processed shape (df_proc): {df_proc.shape} ---")
        if df_proc.empty:
            print("--- GST_CSV DEBUG: CRITICAL - DataFrame is EMPTY after all CSV processing. ---")
        print(f"GST Finder: Processed {len(df_proc)} unique HSN entries from GST CSV.")
        return df_proc
    except Exception as e:
        print(f"GST Finder Error (outer in parse_gst_csv_finder): {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

# --- Model and Data Loading Function (MAIN SETUP) ---
def load_all_resources():
    global donut_processor, donut_model, df_merged_gst_finder, sbert_model_gst_finder, corpus_embeddings_gst_finder, qa_pipeline_gst_finder, all_descriptions_for_fuzzy_gst_finder
    device_for_models = "cuda" if torch.cuda.is_available() else "cpu"; print(f"INFO: Determined device for models: {device_for_models.upper()}")
    print("INFO: Loading DONUT model...");
    try:
        donut_processor = DonutProcessor.from_pretrained(DONUT_MODEL_NAME); donut_model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_NAME)
        donut_model.to(device_for_models); print(f"INFO: DONUT model loaded on {device_for_models.upper()}.")
    except Exception as e: print(f"CRIT ERROR loading DONUT: {e}"); donut_processor, donut_model = None, None

    print("\nINFO: Loading GST Rate Finder resources (SBERT + QA)...")
    try:
        print("--- Load Step 1: Parsing HSN PDF ---"); df_hsn = parse_hsn_pdf_gst_finder(None)
        print(f"--- df_hsn loaded. Shape: {df_hsn.shape} ---")
        print("\n--- Load Step 2: Parsing GST CSV ---"); df_gst = parse_gst_csv_gst_finder(None)
        print(f"--- df_gst loaded. Shape: {df_gst.shape} ---")
        if df_gst.empty: print("CRIT ERROR: df_gst empty. GST Finder cannot work."); df_merged_gst_finder = pd.DataFrame(); return
        print("\n--- Load Step 3: Merging & Combined_Description ---")
        if not df_hsn.empty and 'HS_Code_PDF' in df_hsn.columns: df_hsn['HS_Code_PDF'] = df_hsn['HS_Code_PDF'].astype(str).str.strip()
        if 'HS_Code_GST' in df_gst.columns: df_gst['HS_Code_GST'] = df_gst['HS_Code_GST'].astype(str).str.strip()
        df_hsn_aggregated = pd.DataFrame()
        if not df_hsn.empty: df_hsn_aggregated = aggregate_hsn_descriptions_gst_finder(df_hsn.copy())
        if 'HS_Code_PDF' in df_hsn_aggregated.columns: df_hsn_aggregated['HS_Code_PDF'] = df_hsn_aggregated['HS_Code_PDF'].astype(str).str.strip()
        
        if not df_hsn_aggregated.empty and not df_gst.empty: # Ensure both have data for merge
            print(f"--- Attempting merge: df_gst ({df_gst.shape}) with df_hsn_aggregated ({df_hsn_aggregated.shape}) ---")
            # print(f"--- df_gst HS_Code_GST sample: {df_gst['HS_Code_GST'].head(2).tolist()} ---")
            # print(f"--- df_hsn_aggregated HS_Code_PDF sample: {df_hsn_aggregated['HS_Code_PDF'].head(2).tolist()} ---")
            df_merged_gst_finder = pd.merge(df_gst, df_hsn_aggregated, left_on='HS_Code_GST', right_on='HS_Code_PDF', how='left')
            print(f"--- df_merged_gst_finder AFTER MERGE. Shape: {df_merged_gst_finder.shape} ---")
            if df_merged_gst_finder.empty and not df_gst.empty: print("--- MERGE WARNING: Merge empty, using df_gst as base. ---"); df_merged_gst_finder = df_gst.copy() 
        elif not df_gst.empty : # HSN data was empty, or became empty
            print("--- HSN data (df_hsn_aggregated) is empty or failed. Using df_gst as base for merged data. ---")
            df_merged_gst_finder = df_gst.copy()
        else: # This case should ideally not be hit if df_gst check above worked.
             print("--- CRITICAL: Both df_gst and df_hsn_aggregated are effectively empty before merge. Setting empty df_merged_gst_finder. ---")
             df_merged_gst_finder = pd.DataFrame()


        if not df_merged_gst_finder.empty:
            df_merged_gst_finder.rename(columns={'HS_Code_GST': 'HS_Code'}, inplace=True, errors='ignore')
            if 'HS_Code_PDF' in df_merged_gst_finder.columns and 'HS_Code' in df_merged_gst_finder.columns and 'HS_Code_PDF' != 'HS_Code': df_merged_gst_finder.drop(columns=['HS_Code_PDF'], errors='ignore', inplace=True)

            desc_gst_series = df_merged_gst_finder.get('Description_GST', pd.Series(dtype='str', index=df_merged_gst_finder.index)).fillna('').str.lower().str.strip()
            desc_pdf_agg_series = df_merged_gst_finder.get('Aggregated_Description_PDF', pd.Series(dtype='str', index=df_merged_gst_finder.index)).fillna('').str.lower().str.strip()
            df_merged_gst_finder['Combined_Description'] = desc_pdf_agg_series; mask_pdf_empty = (desc_pdf_agg_series == '')
            df_merged_gst_finder.loc[mask_pdf_empty, 'Combined_Description'] = desc_gst_series[mask_pdf_empty]
            mask_both_valid_and_different = (desc_pdf_agg_series != '') & (desc_gst_series != '') & (desc_pdf_agg_series != desc_gst_series)
            df_merged_gst_finder.loc[mask_both_valid_and_different, 'Combined_Description'] = desc_pdf_agg_series[mask_both_valid_and_different] + " . " + desc_gst_series[mask_both_valid_and_different]
            df_merged_gst_finder['Combined_Description'] = df_merged_gst_finder['Combined_Description'].str.strip()
            if 'HS_Code' in df_merged_gst_finder.columns: df_merged_gst_finder['HS_Code'] = df_merged_gst_finder['HS_Code'].astype(str).str.strip()

        final_cols = ['HS_Code','Combined_Description','Description_GST','Description_PDF','Aggregated_Description_PDF','CGST_Rate','SGST_Rate','IGST_Rate','Is_Compensation_Cess','Compensation_Cess_Rate','Is_Exempted']
        if not df_merged_gst_finder.empty:
            for col in final_cols:
                if col not in df_merged_gst_finder.columns: df_merged_gst_finder[col] = 0.0 if any(k in col for k in ['Rate','Cess']) else (False if 'Is_' in col else None)
            df_merged_gst_finder = df_merged_gst_finder[[c for c in final_cols if c in df_merged_gst_finder.columns]].copy()
        
        print(f"--- df_merged_gst_finder final shape before SBERT: {df_merged_gst_finder.shape} ---")
        # if not df_merged_gst_finder.empty: print("--- DEBUG: df_merged_gst_finder sample:\n" + df_merged_gst_finder[['HS_Code', 'Combined_Description', 'CGST_Rate', 'SGST_Rate', 'IGST_Rate']].head(3).to_string())

        print("\n--- Load Step 4: Preparing SBERT ---")
        if df_merged_gst_finder.empty: print("--- CRIT: df_merged_gst_finder EMPTY before SBERT. ---"); return
        desc_for_sbert = df_merged_gst_finder['Combined_Description'].astype(str).fillna('').tolist()
        if not any(desc_for_sbert): print("WARN: All 'Combined_Description' empty. SBERT will be empty."); sbert_model_gst_finder, corpus_embeddings_gst_finder = None, None
        else:
            sbert_model_gst_finder = SentenceTransformer(SENTENCE_MODEL_NAME_GST_FINDER, device=device_for_models)
            print(f"INFO: SBERT model loaded on {device_for_models.upper()}.")
            corpus_embeddings_gst_finder = sbert_model_gst_finder.encode(desc_for_sbert, convert_to_tensor=True, show_progress_bar=True)
            print(f"INFO: SBERT Corpus Embeddings created on {corpus_embeddings_gst_finder.device if corpus_embeddings_gst_finder is not None else 'N/A'}. Shape: {corpus_embeddings_gst_finder.shape if corpus_embeddings_gst_finder is not None else 'None'}")
        all_descriptions_for_fuzzy_gst_finder = df_merged_gst_finder['Combined_Description'].unique().tolist()

        print("\n--- Load Step 5: Loading QA Pipeline ---")
        qa_pipeline_gst_finder = hf_pipeline('question-answering', model=QA_MODEL_NAME_GST_FINDER, tokenizer=QA_MODEL_NAME_GST_FINDER, device=0 if device_for_models == "cuda" else -1)
        print(f"INFO: QA pipeline loaded on {device_for_models.upper()}.")
    except Exception as e: print(f"CRIT ERROR: Load resources: {e}"); import traceback; traceback.print_exc(); df_merged_gst_finder=pd.DataFrame(); sbert_model_gst_finder,corpus_embeddings_gst_finder,qa_pipeline_gst_finder=None,None,None

# --- OCR and Query Processing Functions ---
def parse_invoice_with_donut(image_path):
    if donut_model is None or donut_processor is None: print("OCR Error: DONUT model/processor not loaded."); return {"error": "OCR Model unavailable."}
    try:
        image = Image.open(image_path).convert("RGB"); pixel_values = donut_processor(image, return_tensors="pt").pixel_values
        task_prompt = "<s_sroie-v1>"; decoder_input_ids = donut_processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        device = "cuda" if torch.cuda.is_available() else "cpu"; pixel_values, decoder_input_ids = pixel_values.to(device), decoder_input_ids.to(device)
        outputs = donut_model.generate(pixel_values,decoder_input_ids=decoder_input_ids,max_length=donut_model.config.decoder.max_position_embeddings,early_stopping=True,pad_token_id=donut_processor.tokenizer.pad_token_id,eos_token_id=donut_processor.tokenizer.eos_token_id,use_cache=True,num_beams=1,bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],return_dict_in_generate=True)
        sequence = donut_processor.batch_decode(outputs.sequences)[0]; sequence = sequence.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
        cleaned_sequence = re.sub(f"^{re.escape(task_prompt)}", "", sequence).strip();
        try:
            parsed_json = donut_processor.token2json(cleaned_sequence);
            # print(f"--- DETAILED DONUT JSON for {os.path.basename(image_path)} ---\n{json.dumps(parsed_json, indent=2)}\n--- END ---");
            return parsed_json
        except Exception as e: print(f"DONUT Error token2json: {e}. Raw: {cleaned_sequence[:200]}..."); return {"error": "Failed to parse OCR to JSON.", "raw_text": cleaned_sequence}
    except Exception as e: print(f"OCR Error: {e}"); import traceback; traceback.print_exc(); return {"error": f"OCR Error: {str(e)}"}

def get_gst_rates_for_product_page_query(user_query):
    global df_merged_gst_finder, sbert_model_gst_finder, corpus_embeddings_gst_finder, qa_pipeline_gst_finder, all_descriptions_for_fuzzy_gst_finder
    if df_merged_gst_finder.empty: return [{"error": "GST data not loaded."}]
    if qa_pipeline_gst_finder is None: return [{"error": "QA model not loaded."}]
    if not user_query or not str(user_query).strip(): return [{"error": "Query is empty."}]
    query_cleaned = str(user_query).lower().strip(); print(f"GST Rate Finder: Processing query: '{query_cleaned}'"); 
    retrieved_item_info = None # Initialize here
    if re.fullmatch(r'\d{2,8}', query_cleaned):
        print(f"INFO: Query '{query_cleaned}' looks like HS Code. Exact match attempt.")
        exact_hs_match_df = df_merged_gst_finder[df_merged_gst_finder['HS_Code'].astype(str).str.strip() == query_cleaned]
        if not exact_hs_match_df.empty: retrieved_item_info = {'row': exact_hs_match_df.iloc[0], 'score': 1.0, 'match_type': 'Exact HS Code'}; print(f"SUCCESS: Found exact HS Code: {query_cleaned}")
    if retrieved_item_info is None and sbert_model_gst_finder is not None and corpus_embeddings_gst_finder is not None and corpus_embeddings_gst_finder.nelement() > 0:
        print(f"INFO: Attempting SBERT search... (Model type: {type(sbert_model_gst_finder)})")
        try:
            q_emb = sbert_model_gst_finder.encode(query_cleaned, convert_to_tensor=True); cos_scores = sbert_util.cos_sim(q_emb, corpus_embeddings_gst_finder)[0]
            if len(cos_scores) > 0:
                top_sbert_res = torch.topk(cos_scores, k=1); s_score, s_idx = top_sbert_res.values[0].item(), top_sbert_res.indices[0].item(); s_thresh = 0.30
                print(f"DEBUG: Top SBERT score: {s_score:.4f} for idx {s_idx}, desc: '{df_merged_gst_finder.iloc[s_idx].get('Combined_Description', '')[:50]}...'")
                if s_score >= s_thresh: retrieved_item_info = {'row': df_merged_gst_finder.iloc[s_idx], 'score': s_score, 'match_type': 'SBERT'}; print(f"SUCCESS: SBERT match score {s_score:.4f}")
                else: print(f"INFO: SBERT score {s_score:.4f} < threshold {s_thresh}.")
            else: print("WARN: SBERT scores empty.")
        except Exception as e: print(f"ERROR: SBERT search failed: {e}"); import traceback; traceback.print_exc()
    if retrieved_item_info is None and all_descriptions_for_fuzzy_gst_finder:
        print("INFO: SBERT inconclusive. Attempting Fuzzy Match..."); best_f = fuzzy_process.extractOne(query_cleaned, all_descriptions_for_fuzzy_gst_finder)
        if best_f:
            f_desc, f_score_100 = best_f[0], best_f[1]; f_score = f_score_100 / 100.0; f_thresh = 0.80
            print(f"DEBUG: Top Fuzzy: '{f_desc[:50]}...' score {f_score:.2f}")
            if f_score >= f_thresh:
                match_rows_df = df_merged_gst_finder[df_merged_gst_finder['Combined_Description'] == f_desc]
                if not match_rows_df.empty: retrieved_item_info = {'row': match_rows_df.iloc[0], 'score': f_score, 'match_type': 'Fuzzy'}; print(f"SUCCESS: Fuzzy match score {f_score:.2f}")
    if retrieved_item_info is None:
        print("INFO: Trying direct keyword search..."); query_words = set(query_cleaned.split()); kw_matches = []
        if query_words:
            for _, row_data in df_merged_gst_finder.iterrows():
                desc_words = set(str(row_data['Combined_Description']).lower().split())
                if query_words.issubset(desc_words): score = (len(query_words)/len(desc_words) if len(desc_words)>0 else 0) + (len(query_words)*0.01);
                if score > 0.2: kw_matches.append({'row':row_data,'score':score,'match_type':'Keyword'})
            if kw_matches: kw_matches.sort(key=lambda x:x['score'],reverse=True); retrieved_item_info=kw_matches[0]; print(f"SUCCESS: Keyword match score {retrieved_item_info['score']:.2f}")
    
    if retrieved_item_info is None: return [{"error": f"No relevant items for '{user_query}'."}]
    best_match_row = retrieved_item_info['row']; match_score = retrieved_item_info['score']; match_type = retrieved_item_info['match_type']
    description_for_qa_context = best_match_row.get('Combined_Description', 'N/A'); retrieved_hs_code_specific = str(best_match_row.get('HS_Code', 'N/A')).strip()
    retrieved_hs_code_specific_cleaned = re.sub(r'\D', '', retrieved_hs_code_specific).lower()
    row_with_rates = pd.Series(dtype='object'); hs_code_used_for_rates = "N/A (Rates not found)"; potential_hs_for_rates = []
    if len(retrieved_hs_code_specific_cleaned) >= 2:
        potential_hs_for_rates.append(retrieved_hs_code_specific_cleaned)
        if len(retrieved_hs_code_specific_cleaned) > 6: potential_hs_for_rates.append(retrieved_hs_code_specific_cleaned[:6])
        if len(retrieved_hs_code_specific_cleaned) > 4: potential_hs_for_rates.append(retrieved_hs_code_specific_cleaned[:4])
        if len(retrieved_hs_code_specific_cleaned) > 2: potential_hs_for_rates.append(retrieved_hs_code_specific_cleaned[:2])
    potential_hs_for_rates = sorted(list(set(potential_hs_for_rates)), key=len, reverse=True)
    print(f"--- DEBUG QA: Rate Lookup: Specific HS is '{retrieved_hs_code_specific_cleaned}'. Potential rate HS codes to check: {potential_hs_for_rates} ---")
    for hs_to_check_for_rates in potential_hs_for_rates:
        if not hs_to_check_for_rates: continue
        temp_df = df_merged_gst_finder[df_merged_gst_finder['HS_Code'].astype(str).str.strip() == hs_to_check_for_rates]
        if not temp_df.empty: row_with_rates = temp_df.iloc[0].copy(); hs_code_used_for_rates = hs_to_check_for_rates; print(f"--- DEBUG QA: Found rates for HS level: {hs_code_used_for_rates} ---"); print(f"--- DEBUG QA: Rate data: CGST={row_with_rates.get('CGST_Rate')}, SGST={row_with_rates.get('SGST_Rate')}, IGST={row_with_rates.get('IGST_Rate')}"); break
    if row_with_rates.empty: print(f"--- DEBUG QA: No rates for HS {retrieved_hs_code_specific} or parents. ---"); default_rate_data = {'CGST_Rate':'unknown','SGST_Rate':'unknown','IGST_Rate':'unknown','Is_Compensation_Cess':False,'Compensation_Cess_Rate':'0','Is_Exempted':False}; row_with_rates = pd.Series(default_rate_data)
    
    qa_context=(f"Product '{description_for_qa_context}' HS {retrieved_hs_code_specific}. CGST {row_with_rates.get('CGST_Rate','unknown')}%. SGST {row_with_rates.get('SGST_Rate','unknown')}%. IGST {row_with_rates.get('IGST_Rate','unknown')}%. "+f"Cess {'applicable' if row_with_rates.get('Is_Compensation_Cess') else 'not applicable'}. Rate {row_with_rates.get('Compensation_Cess_Rate','0')}%. {'Exempted' if row_with_rates.get('Is_Exempted') else 'Not exempted'}.")
    print(f"INFO: QA Context (Type: {match_type}): {description_for_qa_context} (Orig HS: {retrieved_hs_code_specific}), Score: {match_score:.2f}, Rates From: {hs_code_used_for_rates}")
    q_map={"CGST":"CGST?","SGST":"SGST?","IGST":"IGST?","Cess_Applicable":"Cess app?","Cess_Rate":"Cess rate?","Is_Exempted":"Exempted?"}
    qa_dets={}; qa_dbg_list=[f"{match_type} Retrieved (Desc Score: {match_score:.2f}):",f"Prod: {description_for_qa_context} (Orig HS: {retrieved_hs_code_specific})",f"Rates From HS: {hs_code_used_for_rates}",f"DB Rates Found -> CGST: {row_with_rates.get('CGST_Rate','N/A')}, SGST: {row_with_rates.get('SGST_Rate','N/A')}, IGST: {row_with_rates.get('IGST_Rate','N/A')}", "\nQA:"]
    for k,q_txt in q_map.items():
        try:
            res=qa_pipeline_gst_finder({'question':q_txt,'context':qa_context}); ans,conf=res['answer'],res['score']
            if "rate" in q_txt.lower() or k in ["CGST","SGST","IGST","Cess_Rate"]: num_match=re.search(r'(\d+\.?\d*)',ans); ans=num_match.group(1) if num_match else ans
            qa_dets[k]={'answer':ans,'score':conf}; qa_dbg_list.append(f"Q:{q_txt} A:{ans}(Conf:{conf:.2f})")
        except Exception as e: qa_dets[k]={'answer':'QA Error','score':0.0}; qa_dbg_list.append(f"Q:{q_txt} A:Error")
    qa_dbg_info="\n".join(qa_dbg_list)
    cgst,sgst,igst=qa_dets.get('CGST',{}).get('answer','N/A'),qa_dets.get('SGST',{}).get('answer','N/A'),qa_dets.get('IGST',{}).get('answer','N/A')
    cess_app,cess_r,exempt=qa_dets.get('Cess_Applicable',{}).get('answer','N/A').lower(),qa_dets.get('Cess_Rate',{}).get('answer','N/A'),qa_dets.get('Is_Exempted',{}).get('answer','N/A').lower()
    res_template=[{"description_db":description_for_qa_context.capitalize(),"hs_code_db":retrieved_hs_code_specific,
                   "cgst_rate":f"{cgst}%" if cgst not in ['N/A','QA Error'] and re.fullmatch(r'\d+\.?\d*',cgst) else cgst, 
                   "sgst_rate":f"{sgst}%" if sgst not in ['N/A','QA Error'] and re.fullmatch(r'\d+\.?\d*',sgst) else sgst,
                   "igst_rate":f"{igst}%" if igst not in ['N/A','QA Error'] and re.fullmatch(r'\d+\.?\d*',igst) else igst,
                   "cess_applicable":"Yes" if any(k in cess_app for k in ['yes','app']) else ("No" if "no" in cess_app or "not app" in cess_app else "Unk"),
                   "cess_rate":(f"{cess_r}%" if cess_r not in ['N/A','QA Error'] and re.fullmatch(r'\d+\.?\d*',cess_r) else cess_r) if any(k in cess_app for k in ['yes','app']) else "N/A",
                   "is_exempted":"Yes" if any(k in exempt for k in ['yes','exempt']) else ("No" if "no" in exempt or "not ex" in exempt else "Unk"),
                   "retrieval_score":f"{match_score:.2f} ({match_type} on HS {retrieved_hs_code_specific}, Rates from HS {hs_code_used_for_rates})","qa_debug_info":qa_dbg_info}]
    return res_template

# --- Flask Web Routes ---
@app.route('/')
def home(): return render_template('index.html')

@app.route('/reconcile', methods=['GET', 'POST'])
def reconcile_page():
    if request.method == 'POST':
        if 'invoice1' not in request.files or 'invoice2' not in request.files: flash('Both invoice files are required!', 'error'); return redirect(request.url)
        file1, file2 = request.files['invoice1'], request.files['invoice2']
        if not file1.filename or not file2.filename : flash('One or both files not selected!', 'error'); return redirect(request.url)
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            fname1, fname2 = secure_filename(file1.filename), secure_filename(file2.filename)
            uid = uuid.uuid4().hex[:8]; ext1,ext2 = fname1.rsplit('.',1)[1], fname2.rsplit('.',1)[1]
            fpath1, fpath2 = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_inv1.{ext1}"), os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_inv2.{ext2}")
            try:
                file1.save(fpath1); file2.save(fpath2); flash(f'Files uploaded: "{fname1}" & "{fname2}". Processing...', 'info')
                data1, data2 = parse_invoice_with_donut(fpath1), parse_invoice_with_donut(fpath2)
                ocr_errs = []
                if "error" in data1: ocr_errs.append(f"Inv1 OCR: {data1['error']} Raw: {data1.get('raw_text','')[:100]}...")
                if "error" in data2: ocr_errs.append(f"Inv2 OCR: {data2['error']} Raw: {data2.get('raw_text','')[:100]}...")
                if ocr_errs: flash(" ".join(ocr_errs), 'error'); return render_template('reconcile.html', ocr_error1_detail=data1, ocr_error2_detail=data2)
                engine = GSTReconciliationEngine(); comp_res = engine.compare_invoices(data1, data2)
                report_html = engine.generate_reconciliation_report(comp_res, inv1_filename=fname1, inv2_filename=fname2)
                return render_template('reconciliation_report_display.html', report_content=report_html)
            except Exception as e: flash(f'Processing error: {e}', 'error'); import traceback; traceback.print_exc(); return redirect(request.url)
        else: flash('Allowed file types: png, jpg, jpeg, pdf', 'error'); return redirect(request.url)
    return render_template('reconcile.html')

@app.route('/find-gst-rate', methods=['GET', 'POST'])
def find_gst_rate_page():
    results, query = None, "" # Initialize results to None for GET requests
    if request.method == 'POST':
        query = request.form.get('query','').strip()
        if query:
            if df_merged_gst_finder.empty or sbert_model_gst_finder is None or \
               (isinstance(corpus_embeddings_gst_finder, torch.Tensor) and corpus_embeddings_gst_finder.nelement() == 0) or \
               corpus_embeddings_gst_finder is None or \
               qa_pipeline_gst_finder is None:
                 results = [{"error":"GST Rate Finder (SBERT/QA) not fully initialized. Check server logs."}]
            else: 
                results = get_gst_rates_for_product_page_query(query)
        else: 
            results = [{"error":"Please enter a product description."}]
    return render_template('find_gst_rate.html', results=results, query=query)

@app.route('/uploads/<filename>')
def uploaded_file(filename): return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Main Execution Block ---
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER); print(f"INFO: Created '{UPLOAD_FOLDER}'")
    print("INFO: Starting application, loading resources...")
    load_all_resources()
    print("\n--- Resource Loading Sanity Check ---")
    if donut_model is None: print("WARNING: DONUT model (OCR) failed.")
    else: print("INFO: DONUT model loaded.")
    if df_merged_gst_finder.empty: print("WARNING: GST Rate Finder data empty.")
    if sbert_model_gst_finder is None : print("WARNING: SBERT model for GST Rate Finder failed to load.")
    if corpus_embeddings_gst_finder is None or (isinstance(corpus_embeddings_gst_finder, torch.Tensor) and corpus_embeddings_gst_finder.nelement() == 0):
        print("WARNING: SBERT corpus embeddings for GST Rate Finder failed or are empty.")
    if qa_pipeline_gst_finder is None: print("WARNING: QA pipeline for GST Rate Finder failed.")
    
    if not df_merged_gst_finder.empty and \
       sbert_model_gst_finder is not None and \
       corpus_embeddings_gst_finder is not None and corpus_embeddings_gst_finder.nelement() > 0 and \
       qa_pipeline_gst_finder is not None:
        print("INFO: GST Rate Finder (SBERT & QA) resources loaded successfully.")
    else: print("ERROR: Critical resources for GST Rate Finder (SBERT/QA) failed to load or are incomplete.")
    print("--- End Sanity Check ---")
    print("\nINFO: App setup complete. Starting Flask server...")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)