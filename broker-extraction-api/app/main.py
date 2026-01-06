from app.services.position_processor import PositionProcessor
from app.services.transaction_processor import TransactionProcessor
from app.utils import classify_page_type

def process_document(images, yolo_model, ocr_engine):
    pos_processor = PositionProcessor()
    trans_processor = TransactionProcessor()
    
    all_results = []
    
    for img in images:
        ocr_result = ocr_engine.predict(img)
        page_text = " ".join(ocr_result[0]["rec_texts"])
        page_type = classify_page_type(page_text)
        
        if page_type == "position":
            data = pos_processor.process(yolo_model, img, ocr_result)
            all_results.extend(data)
        elif page_type == "transaction":
            data = trans_processor.process(yolo_model, img, ocr_result)
            all_results.extend(data)
            
    return all_results