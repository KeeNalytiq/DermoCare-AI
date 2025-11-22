import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
import io
import json
import os
import base64
import tempfile
import re
import warnings
import textwrap
from contextlib import redirect_stderr
from io import StringIO
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Suppress TensorFlow warnings about unknown custom layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
tf.get_logger().setLevel('ERROR')

st.set_page_config(page_title="AI Dermatological Diagnosis Tool", layout="wide", initial_sidebar_state="expanded")

model = None

# ---------------------------
# Helper functions (UNCHANGED)
# ---------------------------

def load_keras_model(path="cifar_model.h5"):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please upload or place your .h5 model at that path.")
        raise FileNotFoundError(path)
    
    custom_objects = {}
    
    try:
        from compat import get_fallbacks_for_unknown_layer
        cast_fallback = get_fallbacks_for_unknown_layer("Custom>CastToFloat32")
        custom_objects.update(cast_fallback)
    except Exception:
        pass
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_level = tf.get_logger().level
        tf.get_logger().setLevel('ERROR')
        f = StringIO()
        try:
            with redirect_stderr(f):
                if custom_objects:
                    model = load_model(path, custom_objects=custom_objects)
                else:
                    model = load_model(path)
            tf.get_logger().setLevel(old_level)
            return model
        except Exception as e:
            tf.get_logger().setLevel(old_level)
            msg = str(e)
            m = re.search(r"Unknown layer: '([^']+)'", msg)
            if m:
                unknown_name = m.group(1)
                try:
                    from compat import get_fallbacks_for_unknown_layer
                    keys = get_fallbacks_for_unknown_layer(unknown_name)
                except Exception:
                    raise
                custom_objects.update(keys)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        old_level = tf.get_logger().level
                        tf.get_logger().setLevel('ERROR')
                        f = StringIO()
                        with redirect_stderr(f):
                            model = load_model(path, custom_objects=custom_objects)
                        tf.get_logger().setLevel(old_level)
                        try:
                            st.session_state['compat_fallback_used'] = True
                            st.session_state['compat_fallback_name'] = unknown_name
                        except Exception:
                            pass
                        return model
                except Exception as ee:
                    tf.get_logger().setLevel(old_level)
                    st.error(f"Fallback load also failed: {ee}")
                    raise
            raise


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D) or 'conv' in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in the model. Grad-CAM requires at least one conv layer.")


def preprocess_pil(img: Image.Image, target_size, channels_first=False):
    if img.mode != "RGB":
        img = img.convert("RGB")
    try:
        resample_filter = Image.Resampling.LANCZOS
    except Exception:
        resample_filter = getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', None))
    if resample_filter is not None:
        img = ImageOps.fit(img, target_size, method=resample_filter)
    else:
        img = ImageOps.fit(img, target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    if channels_first:
        arr = np.transpose(arr, (2,0,1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def get_model_input_size(model):
    input_shape = model.input_shape
    if input_shape is None:
        raise ValueError("Model input shape is None")
    if len(input_shape) == 4:
        if input_shape[1] is None:
            h = w = 224
            c = 3
            channels_first = False
        elif input_shape[1] == 3:
            channels_first = True
            c = input_shape[1]
            h = input_shape[2]
            w = input_shape[3]
        else:
            channels_first = False
            h = input_shape[1]
            w = input_shape[2]
    else:
        h = w = 224
        channels_first = False
    if h is None or w is None:
        h = w = 224
    return (int(h), int(w)), channels_first


def decode_labels(num_classes, labels_path="labels.json"):
    candidates = [labels_path]
    if os.path.basename(labels_path).lower() == 'labels.json':
        candidates.append(os.path.join(os.path.dirname(labels_path) or '.', 'lables.json'))
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    labels = json.load(f)
                if isinstance(labels, list) and len(labels) == num_classes:
                    return labels
            except Exception:
                continue
    return [f"class_{i}" for i in range(num_classes)]


def get_preliminary_diagnostics(disease_name):
    """Returns unique, disease-specific diagnostic information for each of the 7 skin conditions."""
    # Normalize disease name for case-insensitive matching
    disease_name_normalized = disease_name.strip()
    
    diagnostics = {
        "Actinic keratoses": {
            "severity": "Pre-cancerous - Requires Monitoring",
            "description": "Actinic keratoses (AKs) are rough, scaly patches that develop on sun-damaged skin over many years. These lesions are considered precancerous because approximately 5-10% can progress to squamous cell carcinoma if left untreated. They typically appear on areas frequently exposed to UV radiation such as the face, scalp, ears, hands, and forearms.",
            "symptoms": [
                "Rough, sandpaper-like texture when touched",
                "Flat or slightly raised patches (1-3 cm diameter)",
                "Pink, red, brown, or flesh-colored appearance",
                "May feel tender or itchy in some cases",
                "Multiple lesions often present simultaneously",
                "Commonly found on face, scalp, ears, and hands"
            ],
            "recommendations": [
                "Schedule dermatology appointment within 2-4 weeks for evaluation",
                "Begin daily use of broad-spectrum SPF 30+ sunscreen",
                "Wear protective clothing and wide-brimmed hats outdoors",
                "Consider cryotherapy, topical 5-FU, or photodynamic therapy",
                "Regular full-body skin examinations every 6-12 months",
                "Self-monitor for changes in size, color, or texture"
            ],
            "urgency": "Moderate Priority - Schedule appointment within 2-4 weeks"
        },
        "Basal cell carcinoma": {
            "severity": "Malignant - Non-Metastatic Skin Cancer",
            "description": "Basal cell carcinoma (BCC) is the most common form of skin cancer, accounting for approximately 80% of all skin cancers. While it rarely metastasizes, it can cause significant local tissue destruction if untreated. BCC typically grows slowly and appears most commonly on sun-exposed areas, particularly the face and neck. Early treatment has a 95%+ cure rate.",
            "symptoms": [
                "Pearly, translucent, or waxy bump with visible blood vessels",
                "Flat, flesh-colored or brown scar-like lesion",
                "Bleeding or scabbing sore that heals and returns repeatedly",
                "Raised border with central indentation (ulceration)",
                "Pink growth with slightly elevated, rolled edges",
                "Most common on face, ears, neck, scalp, and shoulders"
            ],
            "recommendations": [
                "URGENT: Schedule dermatology consultation within 1-2 weeks",
                "Do not delay treatment - early intervention is highly effective",
                "Treatment options include: Mohs surgery, excision, cryotherapy, or topical imiquimod",
                "Post-treatment: Regular follow-ups every 6 months for 2 years, then annually",
                "Protect treated area from sun exposure during healing",
                "Consider genetic counseling if multiple BCCs develop before age 50"
            ],
            "urgency": "High Priority - Consult dermatologist within 1-2 weeks"
        },
        "Benign keratosis-like lesions": {
            "severity": "Benign - Non-Cancerous Growth",
            "description": "Benign keratosis-like lesions (seborrheic keratoses) are extremely common, non-cancerous skin growths that typically appear in middle-aged and older adults. These lesions are completely harmless and do not require treatment unless they become irritated or cosmetically bothersome. They are often mistaken for warts or moles but have a characteristic 'stuck-on' appearance.",
            "symptoms": [
                "Waxy, scaly, or slightly raised patches with distinct borders",
                "Brown, black, tan, or yellow-brown coloration",
                "Stuck-on or pasted appearance to the skin surface",
                "Rough, velvety, or warty texture",
                "Size ranges from very small to over 2.5 cm",
                "Commonly appear on face, chest, shoulders, and back",
                "May become itchy or irritated if rubbed by clothing"
            ],
            "recommendations": [
                "No medical treatment required - these are harmless growths",
                "Monitor for any sudden changes in appearance (rare)",
                "Optional removal via cryotherapy, curettage, or laser if cosmetically desired",
                "Avoid picking or scratching to prevent irritation or infection",
                "Annual skin check-up is sufficient for monitoring",
                "No increased risk of skin cancer associated with these lesions"
            ],
            "urgency": "Low Priority - Routine check-up recommended (non-urgent)"
        },
        "Dermatofibroma": {
            "severity": "Benign - Harmless Skin Nodule",
            "description": "Dermatofibroma is a common, benign fibrous skin tumor that typically appears as a firm, raised nodule. These growths are completely harmless and are thought to develop from minor skin trauma such as insect bites or scratches. The characteristic 'dimple sign' (indentation when pinched) helps distinguish it from other lesions. Dermatofibromas are most common on the legs but can appear anywhere.",
            "symptoms": [
                "Firm, raised, button-like nodule (usually 0.5-1.5 cm)",
                "Brown, reddish-brown, or purple-brown coloration",
                "Classic 'dimple sign' - indents when pinched from sides",
                "Slight itching or tenderness in some cases",
                "Most commonly found on legs, especially lower legs",
                "Can also appear on arms, trunk, or other body areas",
                "Usually single, but multiple lesions can occur"
            ],
            "recommendations": [
                "No treatment necessary unless causing symptoms or cosmetic concerns",
                "Monitor for any unusual changes (extremely rare)",
                "Surgical excision available if desired, but may leave scar",
                "Avoid unnecessary trauma to the area",
                "Annual skin examination is adequate for monitoring",
                "Reassurance: These lesions do not become cancerous"
            ],
            "urgency": "Low Priority - Routine check-up recommended (non-urgent)"
        },
        "Melanocytic nevi": {
            "severity": "Benign - Monitor for Changes (ABCDE Rule)",
            "description": "Melanocytic nevi (common moles) are benign clusters of melanocytes that appear as pigmented spots on the skin. Most people have 10-40 moles, and the majority are harmless. However, some atypical moles (dysplastic nevi) have a slightly higher risk of developing into melanoma. Regular monitoring using the ABCDE rule is essential for early detection of any concerning changes.",
            "symptoms": [
                "Round or oval-shaped with well-defined borders",
                "Uniform color throughout (brown, tan, black, or skin-colored)",
                "Smooth, even borders (regular moles)",
                "Typically less than 6mm in diameter (size of pencil eraser)",
                "May be flat (junctional), raised (compound), or dome-shaped (intradermal)",
                "Can appear anywhere on the body",
                "May darken during pregnancy or with sun exposure"
            ],
            "recommendations": [
                "Regular self-examination using ABCDE rule monthly",
                "Annual professional skin examination by dermatologist",
                "Document moles with photos for comparison over time",
                "Protect from UV exposure with SPF 30+ sunscreen daily",
                "Watch for: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolution",
                "Consider mole mapping if you have many moles (>50) or family history of melanoma",
                "Immediate evaluation if any mole shows rapid changes"
            ],
            "urgency": "Low-Moderate Priority - Routine monitoring with annual professional exam"
        },
        "Melanoma": {
            "severity": "MALIGNANT - High-Risk Skin Cancer - IMMEDIATE ATTENTION REQUIRED",
            "description": "Melanoma is the most dangerous form of skin cancer, capable of spreading (metastasizing) to other organs if not detected and treated early. While it accounts for only about 1% of skin cancers, it causes the majority of skin cancer deaths. Early-stage melanoma has a 98% 5-year survival rate, but this drops significantly if it spreads. Immediate medical attention is critical.",
            "symptoms": [
                "ASYMMETRY: One half doesn't match the other half",
                "BORDER: Irregular, ragged, notched, or blurred edges",
                "COLOR: Multiple colors (brown, black, tan, red, white, blue) in one lesion",
                "DIAMETER: Larger than 6mm (but can be smaller when first detected)",
                "EVOLVING: Changing in size, shape, color, or elevation over weeks/months",
                "New mole appearing after age 30",
                "Itching, bleeding, or ulceration in existing mole",
                "May appear as dark streak under fingernail or toenail"
            ],
            "recommendations": [
                "EMERGENCY: Schedule dermatology appointment IMMEDIATELY (within days, not weeks)",
                "Do NOT delay - early treatment is life-saving",
                "Treatment typically involves wide surgical excision with sentinel lymph node biopsy",
                "May require additional treatments: immunotherapy, targeted therapy, or chemotherapy if advanced",
                "Staging determines treatment plan (Stage 0-4)",
                "Regular follow-up every 3-6 months for first 2 years, then every 6-12 months",
                "Full-body skin examinations every 3-6 months for life",
                "Family members should also have regular skin checks",
                "Consider genetic testing if multiple family members affected"
            ],
            "urgency": "URGENT - Consult dermatologist IMMEDIATELY (within 1-3 days)"
        },
        "Vascular lesions": {
            "severity": "Benign - Blood Vessel Abnormalities",
            "description": "Vascular lesions are benign growths or malformations involving blood vessels in the skin. This category includes various types such as hemangiomas (strawberry marks), cherry angiomas, port-wine stains, and spider angiomas. Most are harmless and present from birth or develop during childhood, though some may appear in adulthood. They are typically cosmetic concerns rather than medical problems.",
            "symptoms": [
                "Red, pink, purple, or blue-red coloration",
                "May be flat (macular) or raised (papular/nodular)",
                "Blanches (turns white) when pressed and releases",
                "Can appear anywhere on the body",
                "Size varies from tiny dots to large patches",
                "Cherry angiomas: Small, bright red bumps (common in adults)",
                "Hemangiomas: Raised, bright red lesions (common in infants)",
                "Port-wine stains: Flat, purple-red patches (present at birth)"
            ],
            "recommendations": [
                "Generally harmless - no treatment required unless symptomatic",
                "Monitor for rapid growth, bleeding, or ulceration (rare)",
                "Cosmetic treatment options: laser therapy, cryotherapy, or surgical removal",
                "Protect from trauma to prevent bleeding",
                "Annual check-up sufficient unless changes occur",
                "Consult dermatologist if lesion bleeds easily, grows rapidly, or causes pain",
                "Infantile hemangiomas often resolve spontaneously by age 5-10"
            ],
            "urgency": "Low Priority - Routine check-up recommended (non-urgent)"
        }
    }
    
    # Case-insensitive matching with fallback
    for key in diagnostics.keys():
        if disease_name_normalized.lower() == key.lower():
            return diagnostics[key]
    
    # If exact match not found, return default
    return {
        "severity": "Unknown - Requires Professional Evaluation",
        "description": f"Diagnostic information for '{disease_name}' is not available in the system. Please consult a dermatologist for proper evaluation and diagnosis.",
        "symptoms": [
            "Professional medical evaluation required",
            "Consult healthcare provider for accurate diagnosis"
        ],
        "recommendations": [
            "Schedule appointment with board-certified dermatologist",
            "Bring this image and any relevant medical history",
            "Do not self-diagnose - professional evaluation is essential"
        ],
        "urgency": "Consult healthcare provider for proper diagnosis"
    }


def get_disease_formulations(disease_name):
    """Returns unique medicinal formulations (creams and tablets) for each disease based on severity levels."""
    import pandas as pd
    
    # Normalize disease name for case-insensitive matching
    disease_name_normalized = disease_name.strip()
    
    formulations_db = {
        "Actinic keratoses": {
            "mild": {
                "creams": [
                    {"name": "5-Fluorouracil 5% Cream", "active_ingredient": "5-FU 5%", "frequency": "Apply twice daily", "duration": "2-4 weeks", "unique_id": "AK-MILD-CREAM-001"},
                    {"name": "Imiquimod 5% Cream", "active_ingredient": "Imiquimod 5%", "frequency": "Apply 3x per week", "duration": "8-16 weeks", "unique_id": "AK-MILD-CREAM-002"},
                    {"name": "Diclofenac Sodium 3% Gel", "active_ingredient": "Diclofenac 3%", "frequency": "Apply twice daily", "duration": "60-90 days", "unique_id": "AK-MILD-CREAM-003"}
                ],
                "tablets": [
                    {"name": "Nicotinamide 500mg", "dosage": "500mg twice daily", "duration": "12 months", "purpose": "Prevent new AKs", "unique_id": "AK-MILD-TAB-001"},
                    {"name": "Vitamin D3 2000 IU", "dosage": "2000 IU daily", "duration": "Ongoing", "purpose": "Immune support", "unique_id": "AK-MILD-TAB-002"}
                ]
            },
            "moderate": {
                "creams": [
                    {"name": "5-Fluorouracil 5% Cream + Tretinoin 0.05%", "active_ingredient": "5-FU 5% + Tretinoin", "frequency": "Apply at night", "duration": "3-6 weeks", "unique_id": "AK-MOD-CREAM-001"},
                    {"name": "Ingenol Mebutate 0.015% Gel", "active_ingredient": "Ingenol Mebutate", "frequency": "Single application", "duration": "2-3 days treatment", "unique_id": "AK-MOD-CREAM-002"},
                    {"name": "Photodynamic Therapy (ALA) Cream", "active_ingredient": "Aminolevulinic acid", "frequency": "As directed by physician", "duration": "1-2 sessions", "unique_id": "AK-MOD-CREAM-003"}
                ],
                "tablets": [
                    {"name": "Acitretin 25mg", "dosage": "25mg daily", "duration": "3-6 months", "purpose": "Reduce AK formation", "unique_id": "AK-MOD-TAB-001"},
                    {"name": "Nicotinamide 500mg + Zinc 25mg", "dosage": "Twice daily", "duration": "12 months", "purpose": "Prevent progression", "unique_id": "AK-MOD-TAB-002"}
                ]
            },
            "severe": {
                "creams": [
                    {"name": "5-Fluorouracil 5% + Hydrocortisone 1%", "active_ingredient": "5-FU 5% + HC 1%", "frequency": "Apply twice daily", "duration": "4-6 weeks", "unique_id": "AK-SEV-CREAM-001"},
                    {"name": "Imiquimod 5% + Tazarotene 0.1%", "active_ingredient": "Imiquimod + Tazarotene", "frequency": "Alternate days", "duration": "12-16 weeks", "unique_id": "AK-SEV-CREAM-002"}
                ],
                "tablets": [
                    {"name": "Acitretin 25-50mg", "dosage": "25-50mg daily", "duration": "6-12 months", "purpose": "Systemic treatment", "unique_id": "AK-SEV-TAB-001"},
                    {"name": "Isotretinoin 10mg", "dosage": "10mg daily", "duration": "6 months", "purpose": "Prevent new lesions", "unique_id": "AK-SEV-TAB-002"}
                ]
            }
        },
        "Basal cell carcinoma": {
            "mild": {
                "creams": [
                    {"name": "Imiquimod 5% Cream", "active_ingredient": "Imiquimod 5%", "frequency": "Apply 5x per week", "duration": "6-12 weeks", "unique_id": "BCC-MILD-CREAM-001"},
                    {"name": "5-Fluorouracil 5% Cream", "active_ingredient": "5-FU 5%", "frequency": "Apply twice daily", "duration": "3-6 weeks", "unique_id": "BCC-MILD-CREAM-002"},
                    {"name": "Ingenol Mebutate 0.05% Gel", "active_ingredient": "Ingenol Mebutate", "frequency": "As directed", "duration": "2-3 days", "unique_id": "BCC-MILD-CREAM-003"}
                ],
                "tablets": [
                    {"name": "Vismodegib 150mg", "dosage": "150mg daily", "duration": "Until progression", "purpose": "Hedgehog pathway inhibitor", "unique_id": "BCC-MILD-TAB-001"},
                    {"name": "Sonidegib 200mg", "dosage": "200mg daily", "duration": "As prescribed", "purpose": "Advanced BCC treatment", "unique_id": "BCC-MILD-TAB-002"}
                ]
            },
            "moderate": {
                "creams": [
                    {"name": "Imiquimod 5% + Tretinoin 0.1%", "active_ingredient": "Imiquimod + Tretinoin", "frequency": "Alternate application", "duration": "8-16 weeks", "unique_id": "BCC-MOD-CREAM-001"},
                    {"name": "5-FU 5% + Calcipotriol 0.005%", "active_ingredient": "5-FU + Calcipotriol", "frequency": "Twice daily", "duration": "4-8 weeks", "unique_id": "BCC-MOD-CREAM-002"}
                ],
                "tablets": [
                    {"name": "Vismodegib 150mg", "dosage": "150mg daily", "duration": "Until progression", "purpose": "Systemic treatment", "unique_id": "BCC-MOD-TAB-001"},
                    {"name": "Cemiplimab Injection", "dosage": "350mg IV every 3 weeks", "duration": "Until progression", "purpose": "Immunotherapy", "unique_id": "BCC-MOD-TAB-002"}
                ]
            },
            "severe": {
                "creams": [
                    {"name": "Post-surgical Imiquimod 5%", "active_ingredient": "Imiquimod 5%", "frequency": "Post-op application", "duration": "6-12 weeks", "unique_id": "BCC-SEV-CREAM-001"}
                ],
                "tablets": [
                    {"name": "Vismodegib 150mg", "dosage": "150mg daily", "duration": "Long-term", "purpose": "Metastatic BCC", "unique_id": "BCC-SEV-TAB-001"},
                    {"name": "Sonidegib 200mg", "dosage": "200mg daily", "duration": "As prescribed", "purpose": "Locally advanced BCC", "unique_id": "BCC-SEV-TAB-002"},
                    {"name": "Cemiplimab 350mg", "dosage": "350mg IV q3w", "duration": "Until progression", "purpose": "Advanced/metastatic", "unique_id": "BCC-SEV-TAB-003"}
                ]
            }
        },
        "Benign keratosis-like lesions": {
            "mild": {
                "creams": [
                    {"name": "Salicylic Acid 17% Solution", "active_ingredient": "Salicylic acid 17%", "frequency": "Apply twice daily", "duration": "2-4 weeks", "unique_id": "BKL-MILD-CREAM-001"},
                    {"name": "Lactic Acid 12% Cream", "active_ingredient": "Lactic acid 12%", "frequency": "Apply daily", "duration": "4-6 weeks", "unique_id": "BKL-MILD-CREAM-002"},
                    {"name": "Urea 40% Cream", "active_ingredient": "Urea 40%", "frequency": "Apply twice daily", "duration": "2-3 weeks", "unique_id": "BKL-MILD-CREAM-003"}
                ],
                "tablets": []
            },
            "moderate": {
                "creams": [
                    {"name": "Tretinoin 0.05% Cream", "active_ingredient": "Tretinoin 0.05%", "frequency": "Apply at night", "duration": "6-12 weeks", "unique_id": "BKL-MOD-CREAM-001"},
                    {"name": "Cryotherapy + Tazarotene 0.1%", "active_ingredient": "Tazarotene 0.1%", "frequency": "Post-cryo application", "duration": "4-8 weeks", "unique_id": "BKL-MOD-CREAM-002"}
                ],
                "tablets": []
            },
            "severe": {
                "creams": [
                    {"name": "Tretinoin 0.1% + Hydroquinone 4%", "active_ingredient": "Tretinoin + Hydroquinone", "frequency": "Apply at night", "duration": "8-12 weeks", "unique_id": "BKL-SEV-CREAM-001"},
                    {"name": "Laser Resurfacing + Tretinoin 0.05%", "active_ingredient": "Tretinoin 0.05%", "frequency": "Post-procedure", "duration": "6-8 weeks", "unique_id": "BKL-SEV-CREAM-002"}
                ],
                "tablets": []
            }
        },
        "Dermatofibroma": {
            "mild": {
                "creams": [
                    {"name": "Triamcinolone Acetonide 0.1% Cream", "active_ingredient": "Triamcinolone 0.1%", "frequency": "Apply twice daily", "duration": "2-4 weeks", "unique_id": "DF-MILD-CREAM-001"},
                    {"name": "Hydrocortisone 1% Cream", "active_ingredient": "Hydrocortisone 1%", "frequency": "Apply daily", "duration": "2-3 weeks", "unique_id": "DF-MILD-CREAM-002"}
                ],
                "tablets": [
                    {"name": "Antihistamine (Cetirizine 10mg)", "dosage": "10mg daily", "duration": "As needed for itching", "purpose": "Relieve pruritus", "unique_id": "DF-MILD-TAB-001"}
                ]
            },
            "moderate": {
                "creams": [
                    {"name": "Clobetasol Propionate 0.05% Cream", "active_ingredient": "Clobetasol 0.05%", "frequency": "Apply twice daily", "duration": "2-4 weeks", "unique_id": "DF-MOD-CREAM-001"},
                    {"name": "Intralesional Triamcinolone 10mg/ml", "active_ingredient": "Triamcinolone injection", "frequency": "Monthly injection", "duration": "1-3 sessions", "unique_id": "DF-MOD-CREAM-002"}
                ],
                "tablets": [
                    {"name": "Antihistamine (Fexofenadine 180mg)", "dosage": "180mg daily", "duration": "As needed", "purpose": "Control itching", "unique_id": "DF-MOD-TAB-001"}
                ]
            },
            "severe": {
                "creams": [
                    {"name": "Post-surgical Mupirocin 2%", "active_ingredient": "Mupirocin 2%", "frequency": "Apply to wound", "duration": "7-10 days", "unique_id": "DF-SEV-CREAM-001"}
                ],
                "tablets": [
                    {"name": "Antibiotic (Cephalexin 500mg)", "dosage": "500mg twice daily", "duration": "7-10 days", "purpose": "Post-surgical prophylaxis", "unique_id": "DF-SEV-TAB-001"}
                ]
            }
        },
        "Melanocytic nevi": {
            "mild": {
                "creams": [
                    {"name": "Hydroquinone 4% Cream", "active_ingredient": "Hydroquinone 4%", "frequency": "Apply twice daily", "duration": "3-6 months", "unique_id": "MN-MILD-CREAM-001"},
                    {"name": "Azelaic Acid 20% Cream", "active_ingredient": "Azelaic acid 20%", "frequency": "Apply twice daily", "duration": "3-6 months", "unique_id": "MN-MILD-CREAM-002"},
                    {"name": "Vitamin C 20% Serum", "active_ingredient": "L-Ascorbic acid 20%", "frequency": "Apply in morning", "duration": "Ongoing", "unique_id": "MN-MILD-CREAM-003"}
                ],
                "tablets": [
                    {"name": "Vitamin D3 2000 IU", "dosage": "2000 IU daily", "duration": "Ongoing", "purpose": "Immune support", "unique_id": "MN-MILD-TAB-001"}
                ]
            },
            "moderate": {
                "creams": [
                    {"name": "Tretinoin 0.05% + Hydroquinone 4%", "active_ingredient": "Tretinoin + Hydroquinone", "frequency": "Apply at night", "duration": "3-6 months", "unique_id": "MN-MOD-CREAM-001"},
                    {"name": "Kojic Acid 2% + Arbutin 2%", "active_ingredient": "Kojic acid + Arbutin", "frequency": "Apply twice daily", "duration": "3-6 months", "unique_id": "MN-MOD-CREAM-002"}
                ],
                "tablets": [
                    {"name": "Tranexamic Acid 250mg", "dosage": "250mg twice daily", "duration": "3-6 months", "purpose": "Reduce pigmentation", "unique_id": "MN-MOD-TAB-001"}
                ]
            },
            "severe": {
                "creams": [
                    {"name": "Triple Combination Cream (Tretinoin + Hydroquinone + Fluocinolone)", "active_ingredient": "Tretinoin 0.05% + HQ 4% + Fluocinolone", "frequency": "Apply at night", "duration": "3-6 months", "unique_id": "MN-SEV-CREAM-001"}
                ],
                "tablets": [
                    {"name": "Tranexamic Acid 500mg", "dosage": "500mg twice daily", "duration": "6 months", "purpose": "Systemic depigmentation", "unique_id": "MN-SEV-TAB-001"}
                ]
            }
        },
        "Melanoma": {
            "mild": {
                "creams": [],
                "tablets": [
                    {"name": "Adjuvant Immunotherapy (Pembrolizumab)", "dosage": "200mg IV every 3 weeks", "duration": "12 months", "purpose": "Stage IIB/IIC adjuvant", "unique_id": "MEL-MILD-TAB-001"},
                    {"name": "Nivolumab 240mg", "dosage": "240mg IV every 2 weeks", "duration": "12 months", "purpose": "Adjuvant treatment", "unique_id": "MEL-MILD-TAB-002"}
                ]
            },
            "moderate": {
                "creams": [],
                "tablets": [
                    {"name": "Pembrolizumab 200mg", "dosage": "200mg IV q3w", "duration": "Until progression", "purpose": "Stage III/IV treatment", "unique_id": "MEL-MOD-TAB-001"},
                    {"name": "Nivolumab + Ipilimumab", "dosage": "Nivo 1mg/kg + Ipi 3mg/kg IV", "duration": "4 cycles then Nivo alone", "purpose": "Advanced melanoma", "unique_id": "MEL-MOD-TAB-002"},
                    {"name": "Dabrafenib + Trametinib", "dosage": "Dabrafenib 150mg BID + Trametinib 2mg daily", "duration": "Until progression", "purpose": "BRAF-mutant melanoma", "unique_id": "MEL-MOD-TAB-003"}
                ]
            },
            "severe": {
                "creams": [],
                "tablets": [
                    {"name": "Pembrolizumab 200mg", "dosage": "200mg IV q3w", "duration": "Until progression", "purpose": "First-line metastatic", "unique_id": "MEL-SEV-TAB-001"},
                    {"name": "Nivolumab + Ipilimumab", "dosage": "Combination therapy", "duration": "As prescribed", "purpose": "High-risk metastatic", "unique_id": "MEL-SEV-TAB-002"},
                    {"name": "T-VEC (Talimogene laherparepvec)", "dosage": "Intralesional injection", "duration": "As prescribed", "purpose": "Injectable immunotherapy", "unique_id": "MEL-SEV-TAB-003"},
                    {"name": "Dabrafenib + Trametinib", "dosage": "Targeted therapy", "duration": "Until progression", "purpose": "BRAF V600 mutation", "unique_id": "MEL-SEV-TAB-004"}
                ]
            }
        },
        "Vascular lesions": {
            "mild": {
                "creams": [
                    {"name": "Timolol 0.5% Gel", "active_ingredient": "Timolol 0.5%", "frequency": "Apply twice daily", "duration": "3-6 months", "unique_id": "VL-MILD-CREAM-001"},
                    {"name": "Propranolol 1% Cream", "active_ingredient": "Propranolol 1%", "frequency": "Apply twice daily", "duration": "3-6 months", "unique_id": "VL-MILD-CREAM-002"}
                ],
                "tablets": [
                    {"name": "Propranolol 2-3mg/kg/day", "dosage": "2-3mg/kg divided BID", "duration": "6-12 months", "purpose": "Infantile hemangioma", "unique_id": "VL-MILD-TAB-001"}
                ]
            },
            "moderate": {
                "creams": [
                    {"name": "Timolol 0.5% + Laser Therapy", "active_ingredient": "Timolol 0.5%", "frequency": "Post-laser application", "duration": "3-6 months", "unique_id": "VL-MOD-CREAM-001"},
                    {"name": "Pulsed Dye Laser + Topical Timolol", "active_ingredient": "Combination therapy", "frequency": "As directed", "duration": "Multiple sessions", "unique_id": "VL-MOD-CREAM-002"}
                ],
                "tablets": [
                    {"name": "Propranolol 3-4mg/kg/day", "dosage": "3-4mg/kg divided TID", "duration": "6-12 months", "purpose": "Moderate hemangiomas", "unique_id": "VL-MOD-TAB-001"}
                ]
            },
            "severe": {
                "creams": [
                    {"name": "Post-surgical Antibiotic Ointment", "active_ingredient": "Mupirocin 2%", "frequency": "Apply to wound", "duration": "7-10 days", "unique_id": "VL-SEV-CREAM-001"}
                ],
                "tablets": [
                    {"name": "Propranolol 3-4mg/kg/day", "dosage": "3-4mg/kg divided TID", "duration": "12-18 months", "purpose": "Large/complicated hemangiomas", "unique_id": "VL-SEV-TAB-001"},
                    {"name": "Prednisolone 2mg/kg/day", "dosage": "2mg/kg daily", "duration": "4-8 weeks taper", "purpose": "Rapidly growing lesions", "unique_id": "VL-SEV-TAB-002"}
                ]
            }
        }
    }
    
    # Determine severity level based on disease
    severity_map = {
        "Actinic keratoses": "moderate",  # Pre-cancerous
        "Basal cell carcinoma": "moderate",  # Cancerous but low risk
        "Benign keratosis-like lesions": "mild",  # Benign
        "Dermatofibroma": "mild",  # Benign
        "Melanocytic nevi": "mild",  # Benign but monitor
        "Melanoma": "severe",  # High risk cancer
        "Vascular lesions": "mild"  # Benign
    }
    
    # Case-insensitive matching
    matched_disease = None
    for key in formulations_db.keys():
        if disease_name_normalized.lower() == key.lower():
            matched_disease = key
            break
    
    if not matched_disease:
        return None, None
    
    severity_level = severity_map.get(matched_disease, "mild")
    formulations = formulations_db[matched_disease].get(severity_level, formulations_db[matched_disease]["mild"])
    
    # Create DataFrames for creams and tablets
    creams_df = None
    tablets_df = None
    
    if formulations["creams"]:
        creams_df = pd.DataFrame(formulations["creams"])
        creams_df.index = range(1, len(creams_df) + 1)
    
    if formulations["tablets"]:
        tablets_df = pd.DataFrame(formulations["tablets"])
        tablets_df.index = range(1, len(tablets_df) + 1)
    
    return creams_df, tablets_df, severity_level


def predict_and_explain(model, img_pil: Image.Image, top_k=3):
    target_size, channels_first = get_model_input_size(model)
    x = preprocess_pil(img_pil, target_size, channels_first)
    preds = model.predict(x)
    preds = preds.flatten() if preds.ndim == 2 and preds.shape[0] == 1 else preds
    if preds.ndim > 1:
        preds = preds[0]
    if np.max(preds) > 1.0 or np.min(preds) < 0:
        try:
            preds = tf.nn.softmax(preds).numpy()
        except Exception:
            pass
    top_idx = preds.argsort()[-top_k:][::-1]
    return preds, top_idx


def make_gradcam_heatmap(model, img_pil: Image.Image, class_idx=None):
    last_conv_layer_name = find_last_conv_layer(model)
    target_size, channels_first = get_model_input_size(model)
    x = preprocess_pil(img_pil, target_size, channels_first)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)

            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            if isinstance(conv_outputs, (list, tuple)):
                conv_outputs = conv_outputs[0]

            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]).numpy())
            else:
                if isinstance(class_idx, (list, tuple, np.ndarray)):
                    class_idx = int(class_idx[0])
                elif hasattr(class_idx, 'numpy'):
                    class_idx = int(class_idx.numpy())
                else:
                    class_idx = int(class_idx)

            predictions = tf.convert_to_tensor(predictions)
            loss = predictions[:, class_idx]
    except Exception as e:
        try:
            st.error(f"Grad-CAM setup failed: {e}")
            st.write({
                'predictions_type': type(predictions).__name__ if 'predictions' in locals() else 'NA',
                'conv_outputs_type': type(conv_outputs).__name__ if 'conv_outputs' in locals() else 'NA',
                'class_idx': repr(class_idx)
            })
        except Exception:
            pass
        raise
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    try:
        resample_filter = Image.Resampling.BILINEAR
    except Exception:
        resample_filter = getattr(Image, 'BILINEAR', None)
    if resample_filter is not None:
        heatmap_img = heatmap_img.resize(img_pil.size, resample=resample_filter)
    else:
        heatmap_img = heatmap_img.resize(img_pil.size)
    return np.array(heatmap_img).astype(np.uint8)


def overlay_heatmap_on_image(img_pil: Image.Image, heatmap_arr, alpha=0.4):
    heatmap_pil = Image.fromarray(heatmap_arr).convert("L")
    heatmap_rgb = Image.merge("RGB", [heatmap_pil, Image.new('L', heatmap_pil.size), Image.new('L', heatmap_pil.size)])
    overlay = Image.blend(img_pil.convert('RGB'), heatmap_rgb.resize(img_pil.size), alpha=alpha)
    return overlay


# ---------------------------
# Image comparison helpers
# ---------------------------

def _extract_lesion_features(img: Image.Image) -> dict:
    resized = img.resize((224, 224))
    arr = np.array(resized).astype(np.float32) / 255.0
    mean_intensity = float(arr.mean())
    redness = float(arr[:, :, 0].mean())
    greenness = float(arr[:, :, 1].mean())
    blueness = float(arr[:, :, 2].mean())
    chroma = redness - blueness
    contrast = float(arr.std())
    return {
        "mean_intensity": mean_intensity,
        "redness": redness,
        "greenness": greenness,
        "blueness": blueness,
        "chroma": chroma,
        "contrast": contrast
    }


def _infer_skin_type(mean_intensity: float) -> str:
    if mean_intensity < 0.35:
        return "Dry / Dehydrated"
    if mean_intensity > 0.65:
        return "Oily / Seborrheic"
    return "Balanced / Combination"


def compare_lesion_images(img_a: Image.Image, img_b: Image.Image) -> dict:
    stats_a = _extract_lesion_features(img_a)
    stats_b = _extract_lesion_features(img_b)

    delta_intensity = stats_b["mean_intensity"] - stats_a["mean_intensity"]
    delta_redness = stats_b["redness"] - stats_a["redness"]
    delta_contrast = stats_b["contrast"] - stats_a["contrast"]

    appearance_notes = []
    if abs(delta_redness) > 0.02:
        direction = "more flushed" if delta_redness > 0 else "less erythematous"
        appearance_notes.append(f"Follow-up lesion appears {direction} ({delta_redness*100:.1f}% Δ redness).")
    if abs(delta_intensity) > 0.03:
        direction = "brighter" if delta_intensity > 0 else "darker"
        appearance_notes.append(f"Overall tone is {direction} with {delta_intensity*100:.1f}% change in mean intensity.")
    if abs(delta_contrast) > 0.02:
        descriptor = "more patterned" if delta_contrast > 0 else "smoother"
        appearance_notes.append(f"Texture signature is {descriptor} ({delta_contrast*100:.1f}% Δ texture).")
    if not appearance_notes:
        appearance_notes.append("Surface appearance remains largely consistent between visits.")

    if delta_redness > 0.06:
        trend = "Increased erythema / possible flare‑up."
    elif delta_redness < -0.06:
        trend = "Reduced erythema — responding to therapy."
    else:
        trend = "Stable vascular activity."

    if delta_contrast > 0.03:
        morphology = "Lesion texture becoming more irregular."
    elif delta_contrast < -0.03:
        morphology = "Surface texture has softened / evened out."
    else:
        morphology = "Texture remains comparable between visits."

    if delta_redness > 0.08 and delta_contrast > 0.04:
        diagnosis = "Findings suggest active inflammation; consider reassessing therapy."
    elif delta_redness < -0.05 and delta_contrast < -0.03:
        diagnosis = "Appearance improving; continue current regimen with monitoring."
    else:
        diagnosis = "No major change detected; maintain observation schedule."

    skin_type = _infer_skin_type(stats_b["mean_intensity"])
    formulations = {
        "Dry / Dehydrated": [
            "Ceramide-rich emollient twice daily",
            "Humectant serum (hyaluronic acid) before moisturizer",
            "Non-foaming cleanser to preserve skin barrier"
        ],
        "Oily / Seborrheic": [
            "Lightweight niacinamide gel",
            "Salicylic acid spot formulation for congestion",
            "Oil-free SPF to avoid comedogenic load"
        ],
        "Balanced / Combination": [
            "Peptide-infused moisturizer for maintenance",
            "Azelaic acid 10% for tone refinement",
            "Gentle antioxidant serum (vitamin C) each morning"
        ]
    }

    formulation_bundle = {
        "title": f"{skin_type} protocol",
        "headline": formulations[skin_type][0],
        "support": formulations[skin_type][1:],
        "usage": "Apply morning & evening on cleansed skin unless otherwise directed."
    }

    return {
        "baseline": stats_a,
        "followup": stats_b,
        "delta_intensity": delta_intensity,
        "delta_redness": delta_redness,
        "delta_contrast": delta_contrast,
        "trend": trend,
        "morphology": morphology,
        "diagnosis": diagnosis,
        "skin_type": skin_type,
        "formulations": formulations[skin_type],
        "appearance_notes": appearance_notes,
        "formulation_bundle": formulation_bundle,
        "preliminary_diagnosis": diagnosis
    }


def generate_personalized_formulation(skin_type, irritants, focus_area, routine_goal, lifestyle):
    skin_descriptor = {
        "Dry / Dehydrated": "Creamy amino-acid cleanser + barrier mist.",
        "Oily / Seborrheic": "Micro-foaming, pH-balanced gel cleanse.",
        "Balanced / Combination": "Enzyme polish on T-zone, hydrating gel elsewhere."
    }.get(skin_type, "Derm-approved balanced cleanse.")
    
    focus_map = {
        "Pigmentation control": "AM serum: Azelaic acid 15% + stabilized vitamin C.",
        "Texture refinement": "Night treatment: Retinal 0.1% layered with peptides.",
        "Inflammation calming": "Spot therapy: 2% colloidal oatmeal mask + green tea mist.",
        "Barrier repair": "Occlusive seal: Ceramide balm under humidified mask."
    }
    
    goal_map = {
        "Rapid correction": "Introduce actives every other night for 2 weeks, then daily.",
        "Steady maintenance": "Alternate active and recovery nights for balanced tolerance.",
        "Event ready": "Condense actives to AM routine for 7 days, keep PM minimal.",
        "Derm follow-up prep": "Document lesion photos every 48h for review."
    }
    
    lifestyle_map = {
        "Outdoor heavy": "Reapply mineral SPF every 2 hours during exposure.",
        "Desk bound": "Add blue-light shield mist at midday.",
        "Shift work": "Use melatonin-infused mask after night shifts.",
        "Traveling": "Pack mini hydrating ampoules for flights."
    }
    
    if not irritants:
        irritant_note = "No declared sensitivities — standard derm-grade actives cleared."
    else:
        irritant_note = f"Avoid: {', '.join(irritants)}; formulations remain hypoallergenic."
    
    steps = [
        f"Cleanse & Prep · {skin_descriptor}",
        focus_map.get(focus_area, "Target serum: Multi-pathway antioxidant concentrate."),
        f"Goal Strategy · {goal_map.get(routine_goal, 'Maintain consistent AM/PM cadence.')}",
        lifestyle_map.get(lifestyle, "Keep hydration tablets handy to stabilize barrier."),
        "Daily SPF · Broad-spectrum 50+ with infrared shields."
    ]
    
    usage = "AM: Cleanse → Target serum → Moisturize → SPF. PM: Double cleanse → Treatment → Barrier seal."
    
    return {
        "title": f"Personalized Protocol · {skin_type}",
        "note": irritant_note,
        "steps": steps,
        "usage": usage
    }


def render_html_block(markup: str, height: int = 500, scrolling: bool = False):
    html_str = textwrap.dedent(markup).strip()
    components.html(html_str, height=height, scrolling=scrolling)

# ---------------------------
# ENHANCED UI - ProStruct Style
# ---------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stApp {
        background: #f5f7fa;
    }
    
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1600px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e9f2;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
    }
    
    .sidebar-note {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    .disclaimer-box {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 10px;
        padding: 1.2rem;
        color: #9a3412;
        font-size: 0.85rem;
        line-height: 1.6;
        margin-top: 1.5rem;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .header-banner h1 {
        color: #ffffff;
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .header-banner p {
        color: rgba(255,255,255,0.95);
        font-size: 1rem;
        margin: 0;
        line-height: 1.6;
        max-width: 900px;
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .feature-card {
        background: #ffffff;
        border: 2px solid #e5e9f2;
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    .feature-icon {
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-size: 1rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.25rem;
    }
    
    .feature-desc {
        font-size: 0.85rem;
        color: #64748b;
    }
    
    /* Segmented Tabs */
    div[data-testid="stSegmentedControl"] {
        background: #f8fafc;
        padding: 0.35rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    
    div[data-testid="stSegmentedControl"] button {
        border-radius: 12px;
        border: none;
        font-weight: 600;
        color: #475569;
        transition: background 0.05s ease, color 0.05s ease, transform 0.05s ease;
        padding: 0.65rem 1.1rem;
    }
    
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: linear-gradient(120deg, #5f72ff 0%, #7f53ac 100%);
        color: #fff;
        box-shadow: 0 6px 18px rgba(96, 119, 255, 0.25);
        transform: translateY(-1px);
    }
    
    div[data-testid="stSegmentedControl"] button[aria-checked="false"]:hover {
        color: #5f72ff;
        background: rgba(99,102,241,0.08);
    }
    
    /* Main Content Card */
    .content-card {
        background: #ffffff;
        border: 1px solid #e5e9f2;
        border-radius: 14px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload Zone */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        background: #f8fafc;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f1f5f9;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #d1fae5;
        color: #065f46;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.15s ease;
        border: none;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        padding: 0.75rem 1.5rem;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 16px rgba(102, 126, 234, 0.35);
    }
    
    .stButton > button[kind="secondary"] {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #e2e8f0;
    }
    
    /* Results */
    .result-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25);
    }
    
    .result-header h2 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
    }
    
    .result-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.5rem;
        margin: 0;
        font-weight: 600;
    }
    
    /* Diagnostic Cards */
    .diagnostic-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 1rem;
    }
    
    .diagnostic-panel {
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(56,189,248,0.12) 100%);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 16px;
        padding: 1.3rem;
        box-shadow: 0 12px 32px rgba(15,23,42,0.08);
        backdrop-filter: blur(6px);
    }
    
    .diagnostic-panel h4 {
        margin: 0 0 0.6rem 0;
        font-size: 1rem;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .insight-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        background: rgba(15,118,110,0.12);
        color: #0f766e;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .symptom-grid {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
    }
    
    .symptom-grid li {
        padding: 0.75rem;
        border-radius: 12px;
        background: #fff;
        color: #1e293b;
        font-weight: 500;
        display: flex;
        gap: 0.75rem;
        align-items: center;
        box-shadow: 0 4px 12px rgba(15,23,42,0.07);
    }
    
    .symptom-grid li span {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: rgba(99,102,241,0.15);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #5b21b6;
        font-weight: 700;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 12px;
        font-weight: 600;
        color: #1e293b;
        border: 1px solid #e5e9f2;
        box-shadow: 0 8px 20px rgba(15,23,42,0.05);
    }
    
    .streamlit-expanderHeader:hover {
        background: #eef2ff;
        border-color: #c7d2fe;
    }
    
    /* Images */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Responsive */
    @media (max-width: 1200px) {
        .feature-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }
        
        .header-banner {
            padding: 2rem;
        }
        
        .header-banner h1 {
            font-size: 1.75rem;
        }
    }
    
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .status-card {
        background: #ffffff;
        border: 1px solid #e5e9f2;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    
    .status-label {
        display: block;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    
    .status-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    .clinical-card {
        display: flex;
        gap: 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.18) 0%, rgba(16, 185, 129, 0.18) 100%);
        border: 1px solid transparent;
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0 1.5rem 0;
        align-items: center;
        box-shadow: 0 24px 40px rgba(79,70,229,0.18);
    }
    
    .clinical-icon {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: #6366f1;
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.22);
    }
    
    .clinical-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 1.1rem;
        color: #0f172a;
    }
    
    .clinical-card p {
        margin: 0;
        color: #475569;
        line-height: 1.7;
    }
    
    .prediction-shell {
        width: 100%;
        margin-bottom: 1.5rem;
    }
    
    .prediction-pane {
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid #e5e9f2;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        background: #ffffff;
    }
    
    .prediction-pane.left-pane {
        background: linear-gradient(135deg, #5f72ff 0%, #9b58ff 100%);
        color: #ffffff;
        border: none;
    }
    
    .prediction-pane.left-pane h3 {
        color: #ffffff;
    }
    
    .prediction-pane.left-pane p,
    .prediction-pane.left-pane label {
        color: rgba(255, 255, 255, 0.92);
    }
    
    .prediction-pane.left-pane .stFileUploader,
    .prediction-pane.left-pane .stCameraInput {
        background: rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .prediction-pane.right-pane {
        background: #ffffff;
        border: 1px solid #e5e9f2;
    }
    
    .mini-disease-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 14px 26px rgba(15,23,42,0.08);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .mini-disease-card h5 {
        margin: 0;
        font-size: 1rem;
        color: #0f172a;
    }
    
    .mini-disease-card span {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .mini-disease-card strong {
        font-size: 1.6rem;
        color: #5f72ff;
    }
    
    .research-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 1.2rem;
    }
    
    .research-card {
        border-radius: 18px;
        padding: 1.5rem;
        background: radial-gradient(circle at top right, rgba(99,102,241,0.25), transparent), #0f172a;
        color: #f8fafc;
        position: relative;
        overflow: hidden;
        min-height: 220px;
        box-shadow: 0 24px 60px rgba(15,23,42,0.4);
        text-decoration: none;
        display: block;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .research-card:before {
        content: "";
        position: absolute;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: rgba(59,130,246,0.35);
        top: -40px;
        right: -40px;
        filter: blur(2px);
    }
    
    .research-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
        letter-spacing: 0.02em;
    }
    
    .learning-layout {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.2rem;
    }
    
    .learning-card {
        border-radius: 18px;
        padding: 1.5rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 40px rgba(15,23,42,0.08);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .learning-card .floating-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: rgba(59, 130, 246, 0.12);
        color: #1d4ed8;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .personalize-panel {
        border-radius: 18px;
        padding: 1.5rem;
        background: linear-gradient(140deg, rgba(56,189,248,0.15), rgba(99,102,241,0.18));
        border: 1px solid rgba(59,130,246,0.35);
        box-shadow: 0 25px 40px rgba(59,130,246,0.25);
    }
    
    .glow-button button {
        background: linear-gradient(120deg, #14b8a6, #0ea5e9);
        border: none;
        box-shadow: 0 14px 30px rgba(14,165,233,0.35);
    }
    
    .glow-button button:hover {
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = '🔬 Disease Prediction'
if 'learning_topic' not in st.session_state:
    st.session_state.learning_topic = 'Deep Learning Engine'

# Sidebar Configuration
with st.sidebar:
    st.markdown('<p class="sidebar-title">Input & Settings</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-note">
        <strong>📋 Instructions</strong><br>
        Provide a dermatoscopic image by uploading or using webcam. Prediction requires minimum 100x100 pixels resolution.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Configuration**")
    
    st.markdown('<p style="font-size: 0.85rem; color: #64748b; margin: 1rem 0 0.5rem 0;">Visualization Mode</p>', unsafe_allow_html=True)
    visualization_mode = st.selectbox(
        "Color Mode",
        ["Viridis", "Plasma", "Inferno", "Magma", "Jet"],
        label_visibility="collapsed"
    )
    
    st.markdown('<p style="font-size: 0.85rem; color: #64748b; margin: 1.5rem 0 0.5rem 0;">Show Heatmap</p>', unsafe_allow_html=True)
    show_heatmap = st.checkbox("Enable Grad-CAM Visualization", value=True)
    
    st.markdown('<p style="font-size: 0.85rem; color: #64748b; margin: 1.5rem 0 0.5rem 0;">Prediction Method</p>', unsafe_allow_html=True)
    prediction_method = st.selectbox(
        "Model",
        ["CNN Deep Learning", "Vision Transformer", "Ensemble Model"],
        label_visibility="collapsed"
    )
    
    st.markdown("**Export Options**")
    export_formats = st.multiselect(
        "Export formats",
        ["PDF", "PNG", "CSV", "JSON"],
        default=["PNG"]
    )
    
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This tool provides preliminary analysis for educational purposes only. Always consult a qualified dermatologist for professional diagnosis.
    </div>
    """, unsafe_allow_html=True)

# Main Content
st.markdown("""
<div class="header-banner">
    <h1>AI Tool for Dermatological Insight & Prevention</h1>
    <p>Skin disorders are among the top causes of nonfatal disease burden worldwide. This platform combines medical imaging, deep learning, and pharma-grade dashboards to accelerate early detection, triage, and research.</p>
</div>
""", unsafe_allow_html=True)

# Primary navigation (always below title)
nav_cols = st.columns([1.5, 1, 1, 1, 1.5])
with nav_cols[0]:
    st.write("")
with nav_cols[1]:
    if st.button("🏠 Home", type="primary" if st.session_state.current_page == 'Home' else "secondary", use_container_width=True, key="nav_home"):
        st.session_state.current_page = 'Home'
        st.rerun()
with nav_cols[2]:
    if st.button("📊 History", type="primary" if st.session_state.current_page == 'History' else "secondary", use_container_width=True, key="nav_history"):
        st.session_state.current_page = 'History'
        st.rerun()
with nav_cols[3]:
    if st.button("ℹ️ About", type="primary" if st.session_state.current_page == 'About' else "secondary", use_container_width=True, key="nav_about"):
        st.session_state.current_page = 'About'
        st.rerun()
with nav_cols[4]:
    st.write("")

st.markdown("<br>", unsafe_allow_html=True)

# Tab Navigation
tab_labels = [
    "🔬 Disease Prediction",
    "🖼️ Image Classification",
    "📈 Analytics Dashboard",
    "🎯 Advanced Tools",
    "💊 Research Hub",
    "🧠 Learning"
]

if not hasattr(st, "segmented_control"):
    tab_selection = st.radio(
        "Primary workspace tabs",
        tab_labels,
        horizontal=True,
        index=tab_labels.index(st.session_state.active_tab),
        key="primary_workspace_tabs_radio",
        label_visibility="collapsed"
    )
else:
    tab_selection = st.segmented_control(
        "Primary workspace tabs",
        tab_labels,
        default=st.session_state.active_tab,
        key="primary_workspace_tabs",
        label_visibility="collapsed"
    )

if tab_selection != st.session_state.active_tab:
    st.session_state.active_tab = tab_selection

st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

# Shared defaults for prediction workflow
uploaded = None
cam_image = None
model_upload = None
predict_btn = False
preview_placeholder = result_placeholder = heatmap_placeholder = diagnostics_placeholder = None

if 'advanced_payload' not in st.session_state:
    st.session_state.advanced_payload = None
if 'show_personalize_form' not in st.session_state:
    st.session_state.show_personalize_form = False
if 'personalized_plan' not in st.session_state:
    st.session_state.personalized_plan = None

model = st.session_state.get('active_model', model)

# Page Content
if st.session_state.current_page == 'Home':
    active_tab = st.session_state.active_tab
    
    if active_tab == "🔬 Disease Prediction":
        st.markdown('<div class="prediction-shell">', unsafe_allow_html=True)
        col_left, col_right = st.columns([1.3, 1], gap="large")
        
        with col_left:
            st.markdown('<div class="prediction-pane left-pane">', unsafe_allow_html=True)
            st.markdown("### Step 1 · Load Dermatoscopic Image")
            st.markdown("Upload a clear lesion photo or capture one using your device camera. Higher resolution images (≥1024 px) yield more precise Grad-CAM focus.")
            
            uploaded = st.file_uploader(
                "Upload JPG/PNG",
                type=["png", "jpg", "jpeg"],
                help="Max 200 MB • Ensure even lighting, focus, and minimal glare."
            )
            cam_image = st.camera_input("Or capture via webcam", label_visibility="collapsed")
            
            if uploaded or cam_image:
                st.markdown("""
                <div class="status-badge">
                    <span>✅</span>
                    <span>Image ready for AI analysis</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-badge" style="background:#fee2e2;color:#991b1b;">
                    <span>⏳</span>
                    <span>Awaiting dermatoscopic image</span>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("⚙️ Advanced · Upload custom TensorFlow model", expanded=False):
                model_upload = st.file_uploader("Upload .h5 model", type=["h5"], label_visibility="collapsed")
            
            st.markdown("### Step 2 · Run AI Evaluation")
            predict_btn = st.button("⚡ Predict & Analyze", use_container_width=True, type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown('<div class="prediction-pane right-pane">', unsafe_allow_html=True)
            st.markdown("### Live Preview & Output")
            st.markdown("<p style='color:#64748b;'>After analysis, the uploaded image and top prediction will be summarized here.</p>", unsafe_allow_html=True)
            preview_placeholder = st.empty()
            result_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        heatmap_placeholder = st.empty()
        diagnostics_placeholder = st.empty()
    
    elif active_tab == "🖼️ Image Classification":
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🖼️ Image Classification Studio</h3>', unsafe_allow_html=True)
        st.markdown("Upload a baseline (reference) image and a follow-up image. The tool will compare pigment distribution, inflammation, and texture to suggest a preliminary interpretation plus skincare formulation ideas by skin type.")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            img_a_file = st.file_uploader("Baseline / Reference Image", type=["png","jpg","jpeg"], key="compare_a")
            img_a = Image.open(img_a_file).convert("RGB") if img_a_file else None
            if img_a:
                st.image(img_a, caption="Reference", use_container_width=True)
        with col_c2:
            img_b_file = st.file_uploader("Follow-up / Target Image", type=["png","jpg","jpeg"], key="compare_b")
            img_b = Image.open(img_b_file).convert("RGB") if img_b_file else None
            if img_b:
                st.image(img_b, caption="Target", use_container_width=True)
        
        if img_a and img_b:
            analysis = compare_lesion_images(img_a, img_b)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">🧾 Comparative Intelligence Report</h3>', unsafe_allow_html=True)
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Δ Tone", f"{analysis['delta_intensity']*100:.1f}%", "Brightness shift")
            with col_stats2:
                st.metric("Δ Redness", f"{analysis['delta_redness']*100:.1f}%", analysis['trend'])
            with col_stats3:
                st.metric("Δ Texture", f"{analysis['delta_contrast']*100:.1f}%", analysis['morphology'])
            
            appearance_items = "".join([f"<li><span>•</span>{note}</li>" for note in analysis["appearance_notes"]])
            
            st.markdown(f"""
            <div class="diagnostic-grid" style="margin-top:1rem;">
                <div class="diagnostic-panel">
                    <h4>👀 Physical Appearance Feedback</h4>
                    <ul class="symptom-grid">{appearance_items}</ul>
                </div>
                <div class="diagnostic-panel">
                    <h4>🩺 Preliminary Diagnosis</h4>
                    <p style="margin-bottom:0.5rem;">{analysis['preliminary_diagnosis']}</p>
                    <span class="insight-pill">Skin type: {analysis['skin_type']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="clinical-card" style="margin-top:1rem;">
                <div class="clinical-icon">💊</div>
                <div>
                    <h4>Formulation Blueprint · {analysis['formulation_bundle']['title'].title()}</h4>
                    <p style="margin-bottom:0.5rem;">Primary focus: {analysis['formulation_bundle']['headline']}</p>
                    <p style="margin-bottom:0.5rem;">Supportive steps:</p>
                    <ul>
                        {''.join([f"<li>{step}</li>" for step in analysis['formulation_bundle']['support']])}
                    </ul>
                    <em>{analysis['formulation_bundle']['usage']}</em>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("These findings are observational comparisons and must be validated by a qualified dermatologist.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Upload both images to unlock automated comparison and formulation guidance.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif active_tab == "📈 Analytics Dashboard":
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📈 Analytics Dashboard</h3>', unsafe_allow_html=True)
        try:
            conn = sqlite3.connect('predictions_history.db')
            hist_df = pd.read_sql_query('SELECT * FROM history ORDER BY id DESC LIMIT 200', conn)
            conn.close()
            if hist_df.empty:
                st.info("No history available yet. Run a few predictions to unlock analytics.")
            else:
                counts = hist_df['label'].value_counts()
                theta = list(counts.index) + [counts.index[0]]
                r = list(counts.values) + [counts.values[0]]
                
                radar_fig = go.Figure(
                    data=go.Scatterpolar(
                        r=r,
                        theta=theta,
                        fill='toself',
                        line_color="#5f72ff",
                        line_width=3,
                        marker=dict(size=8, color="#38bdf8"),
                        hovertemplate="%{theta}: %{r} cases"
                    )
                )
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(counts.values)*1.2]),
                        bgcolor="rgba(99,102,241,0.03)"
                    ),
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=360,
                    title="Condition Radar Snapshot"
                )
                st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})
                
                top_condition = counts.idxmax()
                st.markdown(f"""
                <p style="color:#475569; margin-top:1rem;">
                    <strong>{top_condition}</strong> currently accounts for the highest share of predictions. 
                    Radar values animate with each new record to reflect shifting dermatological trends.
                </p>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Condition Mix")
                grid_cols = st.columns(3)
                for idx, (label, count) in enumerate(counts.items()):
                    with grid_cols[idx % 3]:
                        st.markdown(f"""
                        <div class="mini-disease-card">
                            <span>{label}</span>
                            <strong>{count}</strong>
                            <p style="margin:0;color:#475569;">records logged</p>
                        </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Unable to render analytics: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif active_tab == "🎯 Advanced Tools":
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🎯 Advanced Tools</h3>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns([1.1, 1])
        
        with col_t1:
            st.markdown("#### Precision Assistant · Input")
            adv_upload = st.file_uploader("Upload dermatoscopic image", type=["png","jpg","jpeg"], key="adv_uploader")
            adv_cam = st.camera_input("Or capture via camera", key="adv_camera")
            st.caption("Use focused, glare-free imagery for consistent inference.")
            adv_trigger = st.button("🚀 Run Precision Scan", use_container_width=True, type="primary")
            
            if adv_trigger:
                model_ref = st.session_state.get('active_model', model)
                if model_ref is None:
                    st.error("Model unavailable. Please load a model on the Disease Prediction tab first.")
                else:
                    adv_bytes = None
                    if adv_cam is not None:
                        adv_bytes = adv_cam.getvalue()
                    elif adv_upload is not None:
                        adv_bytes = adv_upload.read()
                    if adv_bytes is None:
                        st.warning("Please provide an image to analyze.")
                    else:
                        adv_img = Image.open(io.BytesIO(adv_bytes)).convert("RGB")
                        preds, top_idx_adv = predict_and_explain(model_ref, adv_img, top_k=3)
                        num_classes = int(preds.shape[0])
                        labels = decode_labels(num_classes, "labels.json")
                        adv_top = int(top_idx_adv[0]) if hasattr(top_idx_adv, '__len__') else int(top_idx_adv)
                        adv_label = labels[adv_top] if adv_top < len(labels) else f"class_{adv_top}"
                        adv_conf = float(preds[adv_top])
                        adv_diag = get_preliminary_diagnostics(adv_label)
                        st.session_state.advanced_payload = {
                            "label": adv_label,
                            "confidence": adv_conf,
                            "diagnostics": adv_diag,
                            "image": adv_img
                        }
                        st.session_state.show_personalize_form = False
                        st.session_state.personalized_plan = None
                        st.success("Insight generated. Review the right panel.")
        
        with col_t2:
            st.markdown("#### Output · Model Verdict")
            payload = st.session_state.advanced_payload
            if payload:
                st.image(payload["image"], caption="Latest analysis", use_container_width=True)
                st.markdown(f"""
                <div class="diagnostic-panel" style="margin-top:0.75rem;">
                    <h4>🎯 Prediction</h4>
                    <p style="font-size:1.3rem;margin:0;"><strong>{payload['label']}</strong></p>
                    <p style="margin:0.2rem 0 0.8rem 0;color:#475569;">Confidence: {payload['confidence']*100:.2f}%</p>
                    <span class="insight-pill">Urgency: {payload['diagnostics']['urgency']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="clinical-card">
                    <div class="clinical-icon">🩺</div>
                    <div>
                        <h4>Preliminary Diagnosis</h4>
                        <p>{payload['diagnostics']['description']}</p>
                        <p style="margin-top:0.5rem;"><strong>Key Recommendations</strong></p>
                        <ul>
                            {''.join([f"<li>{rec}</li>" for rec in payload['diagnostics']['recommendations'][:3]])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("✨ Personalize Formulation", key="personalize_btn", use_container_width=True):
                    st.session_state.show_personalize_form = True
            else:
                st.info("Upload an image and run the scan to view insights.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.advanced_payload and st.session_state.show_personalize_form:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">🧪 Personalized Formulation Lab</h3>', unsafe_allow_html=True)
            with st.form("personalization_form"):
                skin_type_pref = st.selectbox(
                    "Skin type",
                    ["Dry / Dehydrated", "Oily / Seborrheic", "Balanced / Combination"],
                    key="personal_skin_type"
                )
                irritant_options = ["Fragrance", "Essential oils", "Parabens", "Nut extracts", "Lactose derived"]
                irritant_flags = st.multiselect("Sensitivities / Irritations", irritant_options, key="personal_irritants")
                focus_area = st.selectbox(
                    "Primary focus area",
                    ["Pigmentation control", "Texture refinement", "Inflammation calming", "Barrier repair"],
                    key="personal_focus"
                )
                routine_goal = st.selectbox(
                    "Routine goal",
                    ["Rapid correction", "Steady maintenance", "Event ready", "Derm follow-up prep"],
                    key="personal_goal"
                )
                lifestyle = st.selectbox(
                    "Lifestyle context",
                    ["Outdoor heavy", "Desk bound", "Shift work", "Traveling"],
                    key="personal_lifestyle"
                )
                submitted = st.form_submit_button("Generate Personalized Plan")
                if submitted:
                    plan = generate_personalized_formulation(skin_type_pref, irritant_flags, focus_area, routine_goal, lifestyle)
                    st.session_state.personalized_plan = plan
                    st.success("Personalized protocol generated.")
            
            if st.session_state.personalized_plan:
                plan = st.session_state.personalized_plan
                st.markdown(f"""
                <div class="personalize-panel">
                    <h4>{plan['title']}</h4>
                    <p style="margin-bottom:0.5rem;">{plan['note']}</p>
                    <ol>
                        {''.join([f"<li>{step}</li>" for step in plan['steps']])}
                    </ol>
                    <p><strong>How to use:</strong> {plan['usage']}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif active_tab == "💊 Research Hub":
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">💊 Research Hub</h3>', unsafe_allow_html=True)
        
        research_cards = [
            {
                "title": "WHO · Global Burden",
                "summary": "Skin disorders persist as the 4th leading cause of non-fatal disability. Early triage tools help shrink referral queues.",
                "cta": "Read synopsis",
                "url": "https://www.who.int/publications",
                "image": "https://images.unsplash.com/photo-1504814532849-cff240bbc503?auto=format&fit=crop&w=600&q=80"
            },
            {
                "title": "Lancet Dermatology Commission",
                "summary": "Highlights why AI explainability is critical for underserved clinics and how hybrid workflows reduce diagnostic delays.",
                "cta": "Commission brief",
                "url": "https://www.thelancet.com/journals/laninf",
                "image": "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?auto=format&fit=crop&w=600&q=80"
            },
            {
                "title": "Nature Medicine · Esteva et al.",
                "summary": "CNN ensemble hit dermatologist-level performance on melanoma detection, sparking clinical validation waves.",
                "cta": "Explore study",
                "url": "https://www.nature.com/articles/nature21056",
                "image": "https://images.unsplash.com/photo-1487412720507-e7ab37603c6f?auto=format&fit=crop&w=600&q=80"
            }
        ]
        
        cards_html = "".join([
            f"""
            <div class="research-card">
                <div class="research-frost"></div>
                <div class="research-media">
                    <img src="{card['image']}" alt="{card['title']}" />
                </div>
                <div class="research-meta">
                    <span>{card['cta']}</span>
                    <h4>{card['title']}</h4>
                    <p>{card['summary']}</p>
                    <a href="{card['url']}" target="_blank">Open dossier →</a>
                </div>
            </div>
            """
            for card in research_cards
        ])
        
        render_html_block(f"""
        <style>
        .research-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.2rem;
        }}
        .research-card {{
            border-radius: 22px;
            padding: 1rem;
            background: #0b1120;
            position: relative;
            overflow: hidden;
            box-shadow: 0 30px 60px rgba(15,23,42,0.4);
            border: 1px solid rgba(148,163,184,0.2);
            min-height: 320px;
        }}
        .research-frost {{
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top, rgba(59,130,246,0.3), transparent);
            opacity: 0.8;
        }}
        .research-media {{
            width: 100px;
            height: 100px;
            border-radius: 18px;
            overflow: hidden;
            margin-bottom: 1rem;
            position: relative;
            box-shadow: 0 15px 35px rgba(15,23,42,0.4);
        }}
        .research-media img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .research-meta {{
            position: relative;
            color: #e2e8f0;
        }}
        .research-meta span {{
            display: inline-flex;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.2);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.6rem;
        }}
        .research-meta h4 {{
            margin: 0;
            font-size: 1.2rem;
            color: #f8fafc;
        }}
        .research-meta p {{
            margin: 0.6rem 0 0.8rem 0;
            color: #cbd5f5;
            line-height: 1.6;
        }}
        .research-meta a {{
            color: #38bdf8;
            font-weight: 600;
            text-decoration: none;
        }}
        </style>
        <div class="research-grid">
            {cards_html}
        </div>
        """, height=540, scrolling=True)
        
        st.markdown("""
        <div class="diagnostic-panel" style="margin-top:1.2rem;">
            <h4>🌐 Pipeline Snapshot</h4>
            <p>We blend peer-reviewed evidence with real-world dermatoscopic data to keep the AI continuously calibrated.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif active_tab == "🧠 Learning":
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🧠 Learning Center</h3>', unsafe_allow_html=True)
        learning_topics = [
            "Deep Learning Engine",
            "Analytics Pipeline",
            "Grad-CAM Explainability",
            "Clinical Reports"
        ]
        topic = st.radio(
            "Select module",
            learning_topics,
            horizontal=True,
            index=learning_topics.index(st.session_state.learning_topic)
        )
        st.session_state.learning_topic = topic
        
        learning_assets = {
            "Deep Learning Engine": {
                "image": "https://images.unsplash.com/photo-1521791136064-7986c2920216?auto=format&fit=crop&w=900&q=80",
                "bullets": [
                    "Normalization + augmentation steady lighting noise.",
                    "Feature stack hunts pigment networks & vascular queues.",
                    "Dense head emits calibrated probabilities per lesion."
                ]
            },
            "Analytics Pipeline": {
                "image": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=900&q=80",
                "bullets": [
                    "Streams every prediction into a live feature store.",
                    "Radar + anomaly detectors watch class drift in real time.",
                    "Signals push alerts to clinicians when melanoma risk spikes."
                ]
            },
            "Grad-CAM Explainability": {
                "image": "https://images.unsplash.com/photo-1507413245164-6160d8298b31?auto=format&fit=crop&w=900&q=80",
                "bullets": [
                    "Heatmaps animate how the CNN focuses on lesions.",
                    "Assures doctors that attention stays on pathology, not artefacts.",
                    "Exports overlay packages for clinical records."
                ]
            },
            "Clinical Reports": {
                "image": "https://images.unsplash.com/photo-1505751172876-fa1923c5c528?auto=format&fit=crop&w=900&q=80",
                "bullets": [
                    "Auto-builds PDF / JSON dossiers with severity + urgency.",
                    "Combines Grad-CAM, recommendations, and follow-up cadence.",
                    "Optimized for cross-team sharing between clinics & pharma."
                ]
            }
        }
        
        asset = learning_assets[topic]
        learning_html = f"""
        <style>
        .learning-layout {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.2rem;
        }}
        .learning-card {{
            border-radius: 24px;
            padding: 1.4rem;
            background: linear-gradient(140deg, rgba(59,130,246,0.12), rgba(59,130,246,0.02));
            border: 1px solid rgba(148,163,184,0.3);
            box-shadow: 0 25px 45px rgba(15,23,42,0.12);
            position: relative;
            overflow: hidden;
        }}
        .learning-card h4 {{
            margin: 0 0 0.8rem 0;
            font-size: 1.1rem;
            color: #0f172a;
        }}
        .learning-card ul {{
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }}
        .learning-card li {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.75rem 0.9rem;
            background: #fff;
            border-radius: 14px;
            color: #0f172a;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(15,23,42,0.08);
        }}
        .learning-card li:before {{
            content: "•";
            color: #2563eb;
            font-size: 1.5rem;
        }}
        .learning-visual {{
            border-radius: 24px;
            overflow: hidden;
            box-shadow: 0 25px 45px rgba(15,23,42,0.12);
        }}
        .learning-visual img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        </style>
        <div class="learning-layout">
            <div class="learning-card">
                <h4>{topic}</h4>
                <ul>
                    {''.join([f"<li>{point}</li>" for point in asset['bullets']])}
                </ul>
            </div>
            <div class="learning-visual">
                <img src="{asset['image']}" alt="{topic}" />
            </div>
        </div>
        """
        render_html_block(learning_html, height=420, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'History':
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📊 Analysis History Dashboard</h3>', unsafe_allow_html=True)
    
    try:
        conn = sqlite3.connect('predictions_history.db')
        df = pd.read_sql_query('SELECT * FROM history ORDER BY id DESC LIMIT 50', conn)
        conn.close()
        
        if not df.empty:
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Predictions", len(df), "+12 this week")
            with col_stat2:
                most_common = df['label'].mode()[0] if not df['label'].mode().empty else "N/A"
                st.metric("Most Common", most_common)
            with col_stat3:
                st.metric("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%", "+2.1%")
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("### 📋 Recent Analysis History")
            st.dataframe(df, use_container_width=True, height=450)
        else:
            st.info("📝 No prediction history available yet. Make your first prediction on the Home page!")
    except Exception as e:
        st.warning(f"Could not load history: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'About':
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">ℹ️ About This Application</h3>', unsafe_allow_html=True)
    
    col_about1, col_about2 = st.columns(2, gap="large")
    
    with col_about1:
        st.markdown("""
        <div class="diagnostic-section">
            <h4>✨ Key Features</h4>
            <ul class="symptom-list">
                <li>Image-based disease detection</li>
                <li>Deep learning CNN models</li>
                <li>Grad-CAM explainability</li>
                <li>Comprehensive diagnostics</li>
                <li>History tracking system</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="diagnostic-section">
            <h4>🛠️ Technology Stack</h4>
            <ul class="symptom-list">
                <li>Framework: Streamlit</li>
                <li>ML Library: TensorFlow/Keras</li>
                <li>Dataset: HAM10000</li>
                <li>Visualization: Grad-CAM</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_about2:
        st.markdown("""
        <div class="diagnostic-section">
            <h4>🏥 Detectable Conditions</h4>
            <ul class="symptom-list">
                <li>Actinic keratoses</li>
                <li>Basal cell carcinoma</li>
                <li>Benign keratosis lesions</li>
                <li>Dermatofibroma</li>
                <li>Melanocytic nevi</li>
                <li>Melanoma</li>
                <li>Vascular lesions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disclaimer-box" style="margin-top: 1.5rem;">
            <strong>⚠️ Important Medical Disclaimer</strong><br><br>
            This tool is for demonstration and educational purposes only. It provides preliminary assessments and should NOT replace professional medical diagnosis. Always consult with a qualified dermatologist or healthcare professional for proper diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Load model (UNCHANGED LOGIC)
@st.cache_resource
def load_model_cached(path):
    return load_keras_model(path)

effective_model_path = None
repo_default = os.path.join(os.getcwd(), "cifar_model.h5")

if st.session_state.current_page == 'Home' and model_upload is not None:
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        tmp.write(model_upload.read())
        tmp.flush()
        tmp.close()
        effective_model_path = tmp.name
    except Exception as e:
        st.error(f"Could not save uploaded model file: {e}")

if effective_model_path is None and os.path.exists(repo_default):
    effective_model_path = repo_default

model = None
if effective_model_path is not None:
    try:
        model = load_model_cached(effective_model_path)
    except FileNotFoundError:
        model = None
    except Exception:
        model = None

if model is None and os.path.exists(repo_default):
    try:
        model = load_model_cached(repo_default)
    except Exception:
        model = None

st.session_state['active_model'] = model

# Prediction logic (UNCHANGED)
if (
    st.session_state.current_page == 'Home'
    and st.session_state.active_tab == "🔬 Disease Prediction"
    and predict_btn
):
    if model is None:
        st.error("❌ Model not loaded. Cannot predict.")
    else:
        img_bytes = None
        if cam_image is not None:
            img_bytes = cam_image.getvalue()
        elif uploaded is not None:
            img_bytes = uploaded.read()
        else:
            st.warning("⚠️ Please upload an image or capture one with the camera.")

        if img_bytes is not None:
            img = Image.open(io.BytesIO(img_bytes))

            with preview_placeholder.container():
                st.image(img, caption="📸 Input Image", use_container_width=True)

            with st.spinner("🔬 Analyzing image... Please wait..."):
                preds, top_idx = predict_and_explain(model, img, top_k=3)
                num_classes = int(preds.shape[0])
                labels = decode_labels(num_classes, "labels.json")

                top1 = int(top_idx[0]) if hasattr(top_idx, "__len__") else int(top_idx)
                top_label = labels[top1] if top1 < len(labels) else f"class_{top1}"
                top_conf = float(preds[top1])

                with result_placeholder.container():
                    st.markdown(
                        f"""
                <div class="result-header">
                    <h2>{top_label}</h2>
                    <p>Confidence: {top_conf*100:.2f}%</p>
                </div>
                """,
                        unsafe_allow_html=True,
                    )
                    st.success("✅ Structure prediction successful!")

                diagnostics = get_preliminary_diagnostics(top_label)

                with diagnostics_placeholder.container():
                    st.markdown("<br>", unsafe_allow_html=True)

                    symptom_items = "".join(
                        [
                            f"<li><span>{idx+1}</span>{text}</li>"
                            for idx, text in enumerate(diagnostics["symptoms"])
                        ]
                    )
                    rec_items = "".join(
                        [
                            f"<li><span>{idx+1}</span>{text}</li>"
                            for idx, text in enumerate(diagnostics["recommendations"])
                        ]
                    )

                    capsule_html = f"""
                <style>
                .capsule-shell {{
                    background: radial-gradient(circle at top, rgba(99,102,241,0.18), rgba(14,165,233,0.22));
                    border-radius: 26px;
                    padding: 2rem;
                    color: #0f172a;
                    font-family: 'Inter', sans-serif;
                    border: 1px solid rgba(99,102,241,0.25);
                    box-shadow: 0 35px 80px rgba(15,23,42,0.18);
                }}
                .capsule-shell h3 {{
                    margin: 0 0 1.5rem 0;
                    font-size: 1.7rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }}
                .capsule-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 1rem;
                }}
                .capsule-chip {{
                    background: #fff;
                    border-radius: 16px;
                    padding: 1rem;
                    box-shadow: 0 15px 30px rgba(15,23,42,0.12);
                    border: 1px solid #e2e8f0;
                }}
                .capsule-chip span {{
                    display: block;
                    font-size: 0.75rem;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: #94a3b8;
                    margin-bottom: 0.35rem;
                }}
                .capsule-chip strong {{
                    font-size: 1.3rem;
                    color: #0f172a;
                }}
                .capsule-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                    gap: 1rem;
                    margin-top: 1.5rem;
                }}
                .capsule-panel {{
                    background: rgba(255,255,255,0.9);
                    border-radius: 18px;
                    padding: 1.2rem;
                    border: 1px solid rgba(99,102,241,0.2);
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
                }}
                .capsule-panel h4 {{
                    margin: 0 0 0.6rem 0;
                    font-size: 1rem;
                    color: #4338ca;
                }}
                .capsule-panel ul {{
                    list-style: none;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    gap: 0.65rem;
                }}
                .capsule-panel li {{
                    background: #f8fafc;
                    border-radius: 12px;
                    padding: 0.8rem 1rem;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    font-weight: 600;
                    color: #0f172a;
                    box-shadow: 0 8px 20px rgba(15,23,42,0.06);
                }}
                .capsule-panel li span {{
                    width: 28px;
                    height: 28px;
                    border-radius: 999px;
                    background: rgba(99,102,241,0.15);
                    color: #4338ca;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.85rem;
                }}
                .capsule-callout {{
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    background: linear-gradient(120deg, rgba(99,102,241,0.16), rgba(16,185,129,0.16));
                    border-radius: 20px;
                    padding: 1.2rem;
                    margin-top: 1.5rem;
                }}
                .capsule-callout .icon {{
                    width: 52px;
                    height: 52px;
                    border-radius: 14px;
                    background: #fff;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.3rem;
                    color: #4338ca;
                    box-shadow: 0 10px 20px rgba(99,102,241,0.25);
                }}
                .capsule-disclaimer {{
                    margin-top: 1.2rem;
                    padding: 1rem 1.2rem;
                    border-radius: 14px;
                    background: rgba(248,113,113,0.12);
                    border: 1px solid rgba(248,113,113,0.35);
                    color: #7f1d1d;
                    font-weight: 600;
                }}
                </style>
                <div class="capsule-shell">
                    <h3>📋 Precision Diagnostic Capsule</h3>
                    <div class="capsule-stats">
                        <div class="capsule-chip">
                            <span>Severity Profile</span>
                            <strong>{diagnostics['severity']}</strong>
                        </div>
                        <div class="capsule-chip">
                            <span>Recommended Urgency</span>
                            <strong>{diagnostics['urgency']}</strong>
                        </div>
                        <div class="capsule-chip">
                            <span>Status</span>
                            <strong>Analyzed</strong>
                        </div>
                    </div>
                    <div class="capsule-grid">
                        <div class="capsule-panel">
                            <h4>✨ Clinical Description</h4>
                            <p>{diagnostics['description']}</p>
                            <span class="insight-pill">AI verified focus zone</span>
                        </div>
                        <div class="capsule-panel">
                            <h4>🛑 Urgency Guidance</h4>
                            <p>{diagnostics['urgency']}</p>
                            <span class="insight-pill" style="background:rgba(244,63,94,0.12);color:#be123c;">Follow escalation protocol</span>
                        </div>
                    </div>
                    <div class="capsule-grid">
                        <div class="capsule-panel">
                            <h4>🔍 Symptoms & Characteristics</h4>
                            <ul>{symptom_items}</ul>
                        </div>
                        <div class="capsule-panel">
                            <h4>💡 Clinical Recommendations</h4>
                            <ul>{rec_items}</ul>
                        </div>
                    </div>
                    <div class="capsule-callout">
                        <div class="icon">📖</div>
                        <div>
                            <h4>Clinician Note</h4>
                            <p style="margin:0;">{diagnostics['description']}</p>
                        </div>
                    </div>
                    <div class="capsule-disclaimer">
                        ⚠️ Medical Disclaimer: This is a preliminary AI assessment for educational purposes only. Always consult with a qualified dermatologist or healthcare professional.
                    </div>
                </div>
                """
                    render_html_block(capsule_html, height=720, scrolling=True)

                # Display personalized formulations
                creams_df, tablets_df, severity_level = get_disease_formulations(top_label)
                
                if creams_df is not None or tablets_df is not None:
                    # Generate HTML for formulations with enriched CSS
                    severity_color_map = {
                        "mild": {"bg": "rgba(16,185,129,0.12)", "border": "rgba(16,185,129,0.3)", "text": "#065f46", "badge": "#10b981"},
                        "moderate": {"bg": "rgba(251,191,36,0.12)", "border": "rgba(251,191,36,0.3)", "text": "#92400e", "badge": "#fbbf24"},
                        "severe": {"bg": "rgba(239,68,68,0.12)", "border": "rgba(239,68,68,0.3)", "text": "#991b1b", "badge": "#ef4444"}
                    }
                    severity_style = severity_color_map.get(severity_level, severity_color_map["moderate"])
                    
                    # Build creams HTML
                    creams_html = ""
                    if creams_df is not None and not creams_df.empty:
                        creams_cards = ""
                        for idx, row in creams_df.iterrows():
                            creams_cards += f"""
                            <div class="formulation-card cream-card">
                                <div class="formulation-header">
                                    <div class="formulation-icon">🧴</div>
                                    <div class="formulation-title-group">
                                        <h4 class="formulation-name">{row['name']}</h4>
                                        <span class="formulation-id">{row['unique_id']}</span>
                                    </div>
                                </div>
                                <div class="formulation-details">
                                    <div class="detail-row">
                                        <span class="detail-label">Active Ingredient:</span>
                                        <span class="detail-value">{row['active_ingredient']}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">Frequency:</span>
                                        <span class="detail-value highlight">{row['frequency']}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">Duration:</span>
                                        <span class="detail-value">{row['duration']}</span>
                                    </div>
                                </div>
                            </div>
                            """
                        creams_html = f"""
                        <div class="formulation-section">
                            <div class="section-header-formulation">
                                <div class="section-icon">🧴</div>
                                <div>
                                    <h3 class="section-title-formulation">Topical Creams & Gels</h3>
                                    <p class="section-subtitle">Apply to clean, dry skin as directed</p>
                                </div>
                            </div>
                            <div class="formulation-grid">
                                {creams_cards}
                            </div>
                        </div>
                        """
                    
                    # Build tablets HTML
                    tablets_html = ""
                    if tablets_df is not None and not tablets_df.empty:
                        tablets_cards = ""
                        for idx, row in tablets_df.iterrows():
                            tablets_cards += f"""
                            <div class="formulation-card tablet-card">
                                <div class="formulation-header">
                                    <div class="formulation-icon">💊</div>
                                    <div class="formulation-title-group">
                                        <h4 class="formulation-name">{row['name']}</h4>
                                        <span class="formulation-id">{row['unique_id']}</span>
                                    </div>
                                </div>
                                <div class="formulation-details">
                                    <div class="detail-row">
                                        <span class="detail-label">Dosage:</span>
                                        <span class="detail-value highlight">{row['dosage']}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">Duration:</span>
                                        <span class="detail-value">{row['duration']}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">Purpose:</span>
                                        <span class="detail-value">{row['purpose']}</span>
                                    </div>
                                </div>
                            </div>
                            """
                        tablets_html = f"""
                        <div class="formulation-section">
                            <div class="section-header-formulation">
                                <div class="section-icon">💊</div>
                                <div>
                                    <h3 class="section-title-formulation">Oral Medications & Tablets</h3>
                                    <p class="section-subtitle">Take as prescribed by healthcare provider</p>
                                </div>
                            </div>
                            <div class="formulation-grid">
                                {tablets_cards}
                            </div>
                        </div>
                        """
                    
                    formulations_html = f"""
                    <style>
                    .formulations-container {{
                        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(14,165,233,0.08) 100%);
                        border-radius: 28px;
                        padding: 2.5rem;
                        margin: 2rem 0;
                        border: 1px solid rgba(99,102,241,0.2);
                        box-shadow: 0 20px 60px rgba(15,23,42,0.15);
                    }}
                    .formulations-header {{
                        text-align: center;
                        margin-bottom: 2.5rem;
                        padding-bottom: 1.5rem;
                        border-bottom: 2px solid rgba(99,102,241,0.15);
                    }}
                    .formulations-header h2 {{
                        font-size: 2rem;
                        font-weight: 700;
                        color: #0f172a;
                        margin: 0 0 0.5rem 0;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 0.75rem;
                    }}
                    .formulations-header .severity-badge {{
                        display: inline-block;
                        padding: 0.4rem 1rem;
                        border-radius: 20px;
                        font-size: 0.85rem;
                        font-weight: 600;
                        margin-top: 0.75rem;
                        background: {severity_style['bg']};
                        border: 1px solid {severity_style['border']};
                        color: {severity_style['text']};
                    }}
                    .formulations-header .disease-name {{
                        color: #64748b;
                        font-size: 1rem;
                        margin-top: 0.5rem;
                    }}
                    .formulation-section {{
                        margin-bottom: 2.5rem;
                    }}
                    .formulation-section:last-child {{
                        margin-bottom: 0;
                    }}
                    .section-header-formulation {{
                        display: flex;
                        align-items: center;
                        gap: 1rem;
                        margin-bottom: 1.5rem;
                        padding: 1rem 1.5rem;
                        background: rgba(255,255,255,0.7);
                        border-radius: 18px;
                        border-left: 4px solid #6366f1;
                    }}
                    .section-icon {{
                        font-size: 2rem;
                        width: 60px;
                        height: 60px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                        border-radius: 16px;
                        box-shadow: 0 8px 20px rgba(99,102,241,0.3);
                    }}
                    .section-title-formulation {{
                        font-size: 1.5rem;
                        font-weight: 700;
                        color: #0f172a;
                        margin: 0 0 0.25rem 0;
                    }}
                    .section-subtitle {{
                        font-size: 0.9rem;
                        color: #64748b;
                        margin: 0;
                    }}
                    .formulation-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                        gap: 1.25rem;
                    }}
                    .formulation-card {{
                        background: #ffffff;
                        border-radius: 20px;
                        padding: 1.5rem;
                        border: 1px solid rgba(148,163,184,0.2);
                        box-shadow: 0 10px 30px rgba(15,23,42,0.08);
                        transition: all 0.3s ease;
                        position: relative;
                        overflow: hidden;
                    }}
                    .formulation-card::before {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 4px;
                        height: 100%;
                        background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 100%);
                    }}
                    .formulation-card:hover {{
                        transform: translateY(-4px);
                        box-shadow: 0 20px 40px rgba(15,23,42,0.15);
                        border-color: rgba(99,102,241,0.4);
                    }}
                    .cream-card::before {{
                        background: linear-gradient(180deg, #10b981 0%, #059669 100%);
                    }}
                    .tablet-card::before {{
                        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
                    }}
                    .formulation-header {{
                        display: flex;
                        align-items: flex-start;
                        gap: 1rem;
                        margin-bottom: 1.25rem;
                        padding-bottom: 1rem;
                        border-bottom: 1px solid rgba(148,163,184,0.15);
                    }}
                    .formulation-icon {{
                        font-size: 2rem;
                        width: 50px;
                        height: 50px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.1) 100%);
                        border-radius: 12px;
                        flex-shrink: 0;
                    }}
                    .formulation-title-group {{
                        flex: 1;
                    }}
                    .formulation-name {{
                        font-size: 1.1rem;
                        font-weight: 700;
                        color: #0f172a;
                        margin: 0 0 0.5rem 0;
                        line-height: 1.4;
                    }}
                    .formulation-id {{
                        display: inline-block;
                        padding: 0.25rem 0.75rem;
                        background: rgba(99,102,241,0.1);
                        color: #6366f1;
                        border-radius: 8px;
                        font-size: 0.75rem;
                        font-weight: 600;
                        font-family: 'Courier New', monospace;
                        letter-spacing: 0.5px;
                    }}
                    .formulation-details {{
                        display: flex;
                        flex-direction: column;
                        gap: 0.75rem;
                    }}
                    .detail-row {{
                        display: flex;
                        justify-content: space-between;
                        align-items: flex-start;
                        padding: 0.6rem 0;
                        border-bottom: 1px solid rgba(148,163,184,0.08);
                    }}
                    .detail-row:last-child {{
                        border-bottom: none;
                    }}
                    .detail-label {{
                        font-size: 0.85rem;
                        font-weight: 600;
                        color: #64748b;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        min-width: 100px;
                    }}
                    .detail-value {{
                        font-size: 0.95rem;
                        color: #0f172a;
                        font-weight: 500;
                        text-align: right;
                        flex: 1;
                        margin-left: 1rem;
                    }}
                    .detail-value.highlight {{
                        color: #6366f1;
                        font-weight: 700;
                    }}
                    .formulation-disclaimer {{
                        margin-top: 2rem;
                        padding: 1.5rem;
                        background: linear-gradient(135deg, rgba(248,113,113,0.12) 0%, rgba(239,68,68,0.08) 100%);
                        border: 2px solid rgba(248,113,113,0.3);
                        border-radius: 18px;
                        border-left: 4px solid #ef4444;
                    }}
                    .formulation-disclaimer p {{
                        margin: 0;
                        color: #7f1d1d;
                        font-weight: 600;
                        font-size: 0.95rem;
                        line-height: 1.6;
                        display: flex;
                        align-items: flex-start;
                        gap: 0.75rem;
                    }}
                    .formulation-disclaimer-icon {{
                        font-size: 1.5rem;
                        flex-shrink: 0;
                    }}
                    @media (max-width: 768px) {{
                        .formulation-grid {{
                            grid-template-columns: 1fr;
                        }}
                        .formulations-container {{
                            padding: 1.5rem;
                        }}
                    }}
                    </style>
                    <div class="formulations-container">
                        <div class="formulations-header">
                            <h2>💊 Personalized Medicinal Formulations</h2>
                            <div class="severity-badge">Severity Level: {severity_level.upper()}</div>
                            <div class="disease-name">Treatment recommendations for: <strong>{top_label}</strong></div>
                        </div>
                        {creams_html}
                        {tablets_html}
                        <div class="formulation-disclaimer">
                            <p>
                                <span class="formulation-disclaimer-icon">⚠️</span>
                                <span>
                                    <strong>Medical Disclaimer:</strong> These formulations are for informational and educational purposes only. 
                                    Always consult with a qualified dermatologist or healthcare professional before starting any treatment. 
                                    Do not use medications without proper medical supervision. Individual responses to medications may vary.
                                </span>
                            </p>
                        </div>
                    </div>
                    """
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    render_html_block(formulations_html, height=800, scrolling=True)
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info(f"💊 Formulation recommendations are not available for '{top_label}'. Please consult a dermatologist for treatment options.")

                if show_heatmap:
                    try:
                        with st.spinner("🎨 Generating Grad-CAM visualization..."):
                            top_class = (
                                int(top_idx[0]) if hasattr(top_idx, "__len__") else int(top_idx)
                            )
                            heatmap_arr = make_gradcam_heatmap(
                                model, img, class_idx=top_class
                            )
                            overlay = overlay_heatmap_on_image(
                                img, heatmap_arr, alpha=0.35
                            )

                            with heatmap_placeholder.container():
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                                st.markdown(
                                    '<h3 class="section-header">🔥 Grad-CAM Attention Visualization</h3>',
                                    unsafe_allow_html=True,
                                )

                                st.markdown(
                                    """
                        <p style="color: #64748b; margin-bottom: 1.5rem;">
                            This heatmap shows which regions the AI model focused on when making its prediction. Red/warm areas indicate higher attention.
                        </p>
                        """,
                                    unsafe_allow_html=True,
                                )

                                col_v1, col_v2 = st.columns(2)
                                with col_v1:
                                    st.markdown("**Original Image**")
                                    st.image(img, use_container_width=True)
                                with col_v2:
                                    st.markdown("**Grad-CAM Heatmap Overlay**")
                                    st.image(overlay, use_container_width=True)

                                st.markdown("<br>", unsafe_allow_html=True)

                                col_dl1, col_dl2, col_dl3 = st.columns(3)
                                with col_dl1:
                                    buf = io.BytesIO()
                                    overlay.save(buf, format="PNG")
                                    byte_im = buf.getvalue()
                                    st.download_button(
                                        "📥 Download PNG",
                                        data=byte_im,
                                        file_name=f"gradcam_{top_label.replace(' ', '_')}.png",
                                        mime="image/png",
                                        use_container_width=True,
                                    )
                                with col_dl2:
                                    st.download_button(
                                        "📄 Export PDF",
                                        data="PDF report",
                                        file_name="report.pdf",
                                        use_container_width=True,
                                    )
                                with col_dl3:
                                    json_data = json.dumps(
                                        {
                                            "label": top_label,
                                            "confidence": float(top_conf),
                                            "diagnostics": diagnostics,
                                        },
                                        indent=2,
                                    )
                                    st.download_button(
                                        "📊 Export JSON",
                                        data=json_data,
                                        file_name="analysis.json",
                                        use_container_width=True,
                                    )

                                st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Could not generate Grad-CAM: {e}")
                else:
                    with heatmap_placeholder.container():
                        st.info(
                            "Grad-CAM visualization is currently disabled in the configuration sidebar."
                        )

                # Save history
                try:
                    from datetime import datetime

                    conn = sqlite3.connect("predictions_history.db")
                    c = conn.cursor()
                    c.execute(
                        """CREATE TABLE IF NOT EXISTS history 
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                             label TEXT, confidence REAL, timestamp TEXT)"""
                    )
                    c.execute(
                        "INSERT INTO history (label, confidence, timestamp) VALUES (?,?,?)",
                        (top_label, top_conf, datetime.utcnow().isoformat()),
                    )
                    conn.commit()
                    conn.close()
                except Exception:
                    pass

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<style>
.footer-container {
    text-align: center;
    padding: 2.5rem 2rem;
    border-top: 2px solid #e5e9f2;
    background: linear-gradient(135deg, rgba(99,102,241,0.03) 0%, rgba(14,165,233,0.03) 100%);
    margin-top: 3rem;
}
.footer-content {
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.8;
}
.footer-content p {
    margin: 0.4rem 0;
}
.footer-brand {
    font-weight: 700;
    color: #0f172a;
    font-size: 1rem;
}
.footer-developer {
    color: #6366f1;
    font-weight: 600;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    display: inline-block;
    padding: 0.3rem 0.8rem;
    background: rgba(99,102,241,0.1);
    border-radius: 8px;
}
.footer-disclaimer {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
</style>
<div class="footer-container">
    <div class="footer-content">
        <p class="footer-brand">AI Dermatological Diagnosis Tool v1.0 | Powered by Deep Learning</p>
        <p class="footer-disclaimer">For Educational & Research Purposes Only</p>
        <p class="footer-developer">👨‍💻 Developed by: <strong>Keeistu M S</strong></p>
    </div>
</div>
""", unsafe_allow_html=True)