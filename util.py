import string
import cv2
import numpy as np
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARACTER MAPPING DICTIONARIES
# ═══════════════════════════════════════════════════════════════════════════════

# Indian plates — common OCR misreads
dict_char_to_int = {
    'O': '0', 'D': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'J': '3',
    'A': '4',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
    'Q': '9',
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'J',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'Q',
}

# US plates — original 7-char format (LL DD LLL)
dict_char_to_int_us = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5',
}

dict_int_to_char_us = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S',
}

# ═══════════════════════════════════════════════════════════════════════════════
#  VALID INDIAN STATE / UT CODES
# ═══════════════════════════════════════════════════════════════════════════════
INDIAN_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML',
    'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN',
    'TR', 'TS', 'UK', 'UP', 'WB', 'AN',
}


# ═══════════════════════════════════════════════════════════════════════════════
#  INDIAN PLATE FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def _is_letter_or_mappable(ch):
    """Check if character is a letter or can be mapped to one."""
    return ch.isalpha() or ch in dict_int_to_char


def _is_digit_or_mappable(ch):
    """Check if character is a digit or can be mapped to one."""
    return ch.isdigit() or ch in dict_char_to_int


def license_complies_format_indian(text):
    """
    Check if text complies with Indian plate format.
    Supports 9-char (1-letter series) and 10-char (2-letter series).
    Format: SS DD S/SS DDDD
    """
    if len(text) not in (9, 10):
        return False

    # Positions 0-1: State code (letters)
    if not _is_letter_or_mappable(text[0]) or not _is_letter_or_mappable(text[1]):
        return False

    # Positions 2-3: District code (digits)
    if not _is_digit_or_mappable(text[2]) or not _is_digit_or_mappable(text[3]):
        return False

    if len(text) == 10:
        # 2-letter series at positions 4-5, digits at 6-9
        if not _is_letter_or_mappable(text[4]) or not _is_letter_or_mappable(text[5]):
            return False
        for i in range(6, 10):
            if not _is_digit_or_mappable(text[i]):
                return False
    else:
        # 1-letter series at position 4, digits at 5-8
        if not _is_letter_or_mappable(text[4]):
            return False
        for i in range(5, 9):
            if not _is_digit_or_mappable(text[i]):
                return False

    return True


def format_license_indian(text):
    """Format Indian plate text by correcting OCR misreads at known positions."""
    result = ''

    if len(text) == 10:
        letter_positions = {0, 1, 4, 5}
        digit_positions = {2, 3, 6, 7, 8, 9}
    else:
        letter_positions = {0, 1, 4}
        digit_positions = {2, 3, 5, 6, 7, 8}

    for j in range(len(text)):
        ch = text[j]
        if j in letter_positions:
            result += dict_int_to_char.get(ch, ch)
        elif j in digit_positions:
            result += dict_char_to_int.get(ch, ch)
        else:
            result += ch

    return result


def validate_indian_state_code(formatted_text):
    """Check if the first 2 characters form a valid Indian state code."""
    if len(formatted_text) >= 2:
        return formatted_text[:2] in INDIAN_STATE_CODES
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  US PLATE FORMAT (7 chars: LL DD LLL)
# ═══════════════════════════════════════════════════════════════════════════════

def license_complies_format_us(text):
    """Check if text complies with US 7-character format: LL DD LLL."""
    if len(text) != 7:
        return False

    for i in [0, 1, 4, 5, 6]:
        if not (text[i] in string.ascii_uppercase or text[i] in dict_int_to_char_us):
            return False
    for i in [2, 3]:
        if not (text[i].isdigit() or text[i] in dict_char_to_int_us):
            return False

    return True


def format_license_us(text):
    """Format US plate text by correcting OCR misreads at known positions."""
    mapping = {
        0: dict_int_to_char_us, 1: dict_int_to_char_us,
        2: dict_char_to_int_us, 3: dict_char_to_int_us,
        4: dict_int_to_char_us, 5: dict_int_to_char_us, 6: dict_int_to_char_us,
    }
    result = ''
    for j in range(7):
        ch = text[j]
        result += mapping[j].get(ch, ch)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-STRATEGY OCR PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_variants(crop_bgr):
    """
    Generate multiple preprocessed versions of the plate crop to maximise
    the chance that EasyOCR reads the text correctly.
    Returns a list of grayscale images to feed to the OCR.
    """
    variants = []

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Original grayscale (no threshold)
    variants.append(gray)

    # 2. Fixed threshold (inverted) — original approach
    _, thresh_inv = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
    variants.append(thresh_inv)

    # 3. Fixed threshold (non-inverted)
    _, thresh_norm = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    variants.append(thresh_norm)

    # 4. Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # 5. Otsu inverted
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(otsu_inv)

    # 6. Adaptive threshold (Gaussian)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)

    # 7. CLAHE enhanced + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    _, clahe_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(clahe_otsu)

    # 8. Bilateral filter + Otsu (noise reduction)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    _, blur_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(blur_otsu)

    return variants


def _clean_ocr_text(raw_text):
    """Strip common noise characters from OCR output."""
    return raw_text.upper().replace(' ', '').replace('-', '').replace('.', '').replace(':', '').replace("'", '').replace('"', '').replace('[', '').replace(']', '').replace('(', '').replace(')', '')


def _try_match(text):
    """
    Try to match text against Indian then US format.
    Returns (formatted_text, format_type) or (None, None).
    """
    # Indian format
    if license_complies_format_indian(text):
        formatted = format_license_indian(text)
        if validate_indian_state_code(formatted):
            return formatted, 'indian'

    # US format
    if license_complies_format_us(text):
        formatted = format_license_us(text)
        return formatted, 'us'

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN READ FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def read_license_plate(license_plate_crop_bgr):
    """
    Read the license plate text from a BGR crop image.

    Tries multiple preprocessing strategies and returns the best match
    that conforms to either Indian or US plate format.

    Returns:
        tuple: (formatted_plate_text, confidence_score) or (None, None).
    """
    variants = _preprocess_variants(license_plate_crop_bgr)

    best_result = None
    best_score = -1.0

    for img in variants:
        try:
            detections = reader.readtext(img)
        except Exception:
            continue

        if not detections:
            continue

        # Combine all fragments from this variant
        full_text = ''
        total_score = 0.0
        count = 0
        for _, text, score in detections:
            full_text += _clean_ocr_text(text)
            total_score += score
            count += 1

        if count == 0 or len(full_text) < 4:
            continue

        avg_score = total_score / count

        # Try matching the combined text
        matched, fmt = _try_match(full_text)
        if matched and avg_score > best_score:
            best_result = matched
            best_score = avg_score

        # Also try individual detections (sometimes a single fragment is the plate)
        for _, text, score in detections:
            clean = _clean_ocr_text(text)
            if len(clean) < 4:
                continue
            matched, fmt = _try_match(clean)
            if matched and score > best_score:
                best_result = matched
                best_score = score

    if best_result is not None:
        return best_result, best_score

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  VEHICLE MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their coordinates.

    Returns:
        tuple: (x1, y1, x2, y2, car_id) or (-1, -1, -1, -1, -1) if not found.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV WRITER (legacy — kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(results, output_path):
    """
    Write frame-level results to a CSV file.
    """
    import os
    try:
        f = open(output_path, 'w')
    except PermissionError:
        base, ext = os.path.splitext(output_path)
        counter = 1
        while True:
            new_path = f"{base}_{counter}{ext}"
            try:
                f = open(new_path, 'w')
                print(f"\n[WARNING] Permission denied to write to '{output_path}'.")
                print(f"[WARNING] Saved to fallback path: '{new_path}' instead.\n")
                break
            except PermissionError:
                counter += 1
                if counter > 100:
                    raise

    with f:
        f.write('{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                entry = results[frame_nmr][car_id]
                if 'car' in entry and 'license_plate' in entry and 'text' in entry['license_plate']:
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr, car_id,
                        '[{} {} {} {}]'.format(*entry['car']['bbox']),
                        '[{} {} {} {}]'.format(*entry['license_plate']['bbox']),
                        entry['license_plate']['bbox_score'],
                        entry['license_plate']['text'],
                        entry['license_plate']['text_score']))