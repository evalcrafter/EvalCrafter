dir_videos='./videos'

# [need for speicific platform] pip install spatial_correlation_sampler
# pip install spatial_correlation_sampler==0.4.0 --index-url https://mirrors.tencent.com/pypi/simple/

# Celebrity ID Score
cd metrics/deepface
python3 celebrity_id_score.py --dir_videos  $dir_videos

# IS
cd ..
python3 is.py --dir_videos $dir_videos  &

# # OCR Score
python3 ocr_score.py --dir_videos $dir_videos --metric 'ocr_score'

# # VQA_A and VQA_T
cd DOVER
python3 evaluate_a_set_of_videos.py --dir_videos $dir_videos

cd ..
cd Scores_with_CLIP 
CLIP-Score 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_score'
# Face Consistency 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'face_consistency_score'
# SD-Score 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'sd_score'
# BLIP-BLUE 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'blip_bleu'
# CLIP-Temp 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_temp_score'

# # # Action Score
cd ..
cd mmaction2/demo
python3 action_score.py --dir_videos $dir_videos --metric 'action_score'

cd ..
cd ..
cd RAFT
# Flow-Score
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'flow_score' &
# Motion AC-Score
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'motion_ac_score'
# Warping Error
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'warping_error' 

cd ..
cd Segment-and-Track-Anything
# Count-Score
python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'count_score' &
# # Color-Score
python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'color_score' 
# Detection-Score
python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'detection_score'


# # Final results
cd ..
cd ..
python eval_from_metrics.py 


