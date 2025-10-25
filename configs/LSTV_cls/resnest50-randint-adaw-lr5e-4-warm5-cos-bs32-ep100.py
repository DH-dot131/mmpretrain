_base_ = [
    '../_base_/models/lstv_resnest50.py', '../_base_/datasets/lstv_LAT_bs32.py',
    '../_base_/schedules/lstv_bs256.py', '../_base_/default_runtime.py'
]


import os, inspect

# 현재 실행 중인 config 파일 경로
_config_path = inspect.getfile(inspect.currentframe())
# 파일명만 추출 (확장자 .py 제외)
_cfg_name = os.path.splitext(os.path.basename(_config_path))[0]

# work_dir에 동적으로 추가
work_dir = os.path.join(
    '..', 'work_dirs', 'lstv_classification_v2', _cfg_name
)



auto_scale_lr = dict(base_batch_size=256)

val_evaluator = [dict(type='Accuracy', thrs=0.5, collect_device='gpu'),
                dict(type='SingleLabelMetric', 
                     thrs = 0.5, 
                     num_classes=2, 
                     average = None,
                     collect_device = 'gpu'
                     ),
                 dict(type='ConfusionMatrix',
                     num_classes=2,
                     collect_device='gpu'
                     ),
                 dict(out_file_path= work_dir + '/val_results.pkl', type='DumpResults')]

test_evaluator =[dict(type='Accuracy', thrs=0.5, collect_device='gpu'),
                dict(type='SingleLabelMetric', 
                     thrs = 0.5, 
                     num_classes=2, 
                     average = None,
                     collect_device = 'gpu'
                     ),
                 dict(type='ConfusionMatrix',
                     num_classes=2,
                     collect_device='gpu'
                     ),
                 dict(out_file_path= work_dir + '/test_results.pkl', type='DumpResults')] 
