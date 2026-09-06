"""Запуск из корня проекта: python -m app.app."""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
import gradio as gr
from app.preprocessing import preprocess

BASE=Path(__file__).resolve().parent
LABELS={'butterfly':'Бабочка','dragonfly':'Стрекоза','ladybird':'Божья коровка'}


def build_app():
    metadata=json.loads((BASE/'model.json').read_text(encoding='utf-8'))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(BASE/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    def predict(image):
        if image is None:return None
        x=preprocess(image,metadata['config'])[None]
        logits=session.run(None,{'image':x})[0][0]
        probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum()
        return {LABELS.get(label,label):float(p) for label,p in zip(metadata['classes'],probabilities)}
    interface=gr.Interface(predict,inputs=gr.Image(type='pil',label='Изображение'),
                   outputs=gr.Label(label='Результат',num_top_classes=3),title='Три вида насекомых',
                   description='Загрузите фото бабочки, стрекозы или божьей коровки. '
                   'Модель выбирает один из этих трёх видов; другие виды она не определяет.',
                   flagging_mode='never')
    return interface


if __name__=='__main__':
    build_app().launch(server_name='127.0.0.1',server_port=7860,share=False)
