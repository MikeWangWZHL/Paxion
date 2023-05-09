## Model version tracker

### Ours
- `internvideo_train`: original dual perceiver implementation with forward/backward/inverse dynamic loss
- `internvideo_train_v2_use_attn_guidance`: internvideo_train + simple attention guidance 
- `internvideo_train_inverse_dynamic_only`: single perceiver (knowledge_perceiver) with inverse dynamic loss
- `internvideo_train_v3_full`: using unconditional query and momentum knowledge_perceiver for generating targets in forward/backward dynamic; adding L2 regularization for the first/second half querys in forward/backward dynamic; 
  - objectives: ["inverse_dynamic","forward_dynamic","backward_dynamic","state_change_regularization"]
  - loss_weighting: [1.0,1.0,1.0,1.0]
  - if_use_dual_perceiver: True
  - if_use_momentum: True
- `internvideo_train_v3_single`: removing query perceiver from internvideo_train_v3_full
  - objectives: ["inverse_dynamic","forward_dynamic","backward_dynamic","state_change_regularization"]
  - loss_weighting: [1.0,1.0,1.0,1.0]
  - if_use_dual_perceiver: False
  - if_use_momentum: True
<!-- - `internvideo_train_v3_single_no_momentum`: removing query perceiver from internvideo_train_v3_full
  - objectives: ["inverse_dynamic","forward_dynamic","backward_dynamic","state_change_regularization"]
  - loss_weighting: [1.0,1.0,1.0,1.0]
  - if_use_dual_perceiver: False
  - if_use_momentum: False -->
- `internvideo_train_v3_no_reg`: removing "state_change_regularization" from internvideo_train_v3_full; 
  - objectives: ["inverse_dynamic","forward_dynamic","backward_dynamic"]
  - loss_weighting: [1.0,1.0,1.0]
  - if_use_dual_perceiver: True
  - if_use_momentum: True
- `internvideo_train_v3_no_momentum`: removing momentum usage from internvideo_train_v3_full; 
  - objectives: ["inverse_dynamic","forward_dynamic","backward_dynamic","state_change_regularization"]
  - loss_weighting: [1.0,1.0,1.0,1.0]
  - if_use_dual_perceiver: True
  - if_use_momentum: False

### Baseline
- `internvideo_baseline_train`: adding text and vision transformer adaptor
- `internvideo_baseline_train_simple`: adding only vision transformer adaptor
