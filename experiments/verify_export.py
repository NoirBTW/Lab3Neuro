from pathlib import Path
import csv,json,time
import numpy as np
import torch,timm,onnxruntime as ort
from PIL import Image
from app.preprocessing import preprocess
from experiments.train import ROOT,TrainConfig,Images

def verify(checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=TrainConfig(**saved['config']);torch.set_num_threads(cfg.threads)
    model=timm.create_model(cfg.model,pretrained=False,num_classes=len(saved['classes']))
    model.load_state_dict(saved['state_dict']);model.eval()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/'app/model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    rows=list(csv.DictReader((ROOT/cfg.manifest).open(encoding='utf-8')))
    dataset=Images(rows,'test',saved['classes'],cfg)
    batch=np.stack([preprocess(Image.open(ROOT/r['path']),saved['config']) for r in dataset.rows])
    for i in range(len(dataset)):np.testing.assert_array_equal(dataset[i][0].numpy(),batch[i])
    with torch.no_grad():expected=model(torch.from_numpy(batch)).numpy()
    actual=session.run(None,{'image':batch})[0]
    np.testing.assert_allclose(expected,actual,atol=1e-4,rtol=1e-3)
    np.testing.assert_array_equal(expected.argmax(1),actual.argmax(1))
    single=batch[:1]
    for _ in range(10):session.run(None,{'image':single})
    latencies=[]
    for _ in range(50):
        start=time.perf_counter();session.run(None,{'image':single});latencies.append((time.perf_counter()-start)*1000)
    result=dict(test_images=len(batch),max_logit_error=float(abs(expected-actual).max()),
                predictions_equal=True,preprocessing_equal=True,onnx_megabytes=(ROOT/'app/model.onnx').stat().st_size/1e6,
                cpu_median_ms=float(np.median(latencies)),cpu_p95_ms=float(np.percentile(latencies,95)),
                timing='batch 1, two CPU threads, 10 warmup + 50 runs, preprocessing excluded')
    (ROOT/'experiments/reports/export_verification.json').write_text(json.dumps(result,indent=2))
    return result
