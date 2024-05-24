# evaluate on sintel
python evaluate.py --cfg config/eval/sintel-M.json --model models/Tartan-C-T-TSKH432x960-M.pth
# evaluate on kitti
python evaluate.py --cfg config/eval/kitti-M.json --model models/Tartan-C-T-TSKH-kitti432x960-M.pth
# evaluate on spring
python evaluate.py --cfg config/eval/spring-M.json --model models/Tartan-C-T-TSKH-spring540x960-M.pth