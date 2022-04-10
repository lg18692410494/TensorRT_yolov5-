#请先配置好pip环境和python环境
pip install trtpy
python3 -m trtpy get-env

python3 -m trtpy get-templ cpp-simple-yolov5
cd cpp-simple-yolov5
make run
#一键配置，一键运行，遇到问题可加vx