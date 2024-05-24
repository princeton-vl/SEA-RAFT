# submit to sintel public leaderboard
python submission.py --cfg config/eval/sintel-M.json --model models/Tartan-C-T-TSKH432x960-M.pth
# submit to kitti public leaderboard
python submission.py --cfg config/eval/kitti-M.json --model models/Tartan-C-T-TSKH-kitti432x960-M.pth
# submit to spring public leaderboard
python submission.py --cfg config/eval/spring-M.json --model models/Tartan-C-T-TSKH-spring540x960-M.pth