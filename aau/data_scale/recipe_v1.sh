# Ricetta v1 del congiunto, flag per flag come aau/cross3dmm/train_joint_E.sbatch (braccio ctrl),
# senza epoche, seme, eval_every, save_every, dati e run dir, che mette chi la usa.
#   source aau/data_scale/recipe_v1.sh; "${RECIPE_V1[@]}"
RECIPE_V1=(
  --device cuda --model xyz_dn
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300
  --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0
  --train_level mixed --train_pair_mode cross_topology
  --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5
  --use_id_loss --lambda_id 0.25
  --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7
  --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6
  --max_subjects_eval_train 16 --preload_eval_workers_train 2
  --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2
  --noise_modes translation,rotation,jitter
  --noise_mode_weights translation=4,rotation=2,jitter=1
  --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5
  --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001
  --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6
  --eval_mode average
)
