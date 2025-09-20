.PHONY: 
.ONESHELL:

all:

build-flp:
	cd FasterLivePortrait
	docker build -t flp_api:init .
	docker rm -f flp_api || true
	docker run -it --name=flp_api --gpus=all flp_api:init bash -ic "/root/FasterLivePortrait/build_trt.sh"
	docker commit flp_api flp_api:trt
	docker rm -f flp_api

update-flp-api:
	cd FasterLivePortrait
	docker rm -f flp_api_update || true
	docker run -it -d --name=flp_api_update flp_api:trt bash
	docker cp api_v2.py flp_api_update:/root/FasterLivePortrait/api_v2.py
	docker commit flp_api_update flp_api:trt
	docker rm -f flp_api_update
	
build-tts:
	cd TTS
	docker build -t f5_tts_api .