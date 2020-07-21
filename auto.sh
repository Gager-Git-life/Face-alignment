#!/bin/sh

if [ ! -d "./build/" ];then
  echo "[INFO]>>> 创建build文件夹"
  mkdir ./build
else
  echo "[INFO]>>> 清空build下内容"
  rm -rf build/*
fi

cd build
cmake ..
make -j8
mv Ultra-face-mnn ..
cd ..

if [ $1 = "pic" ]
then
	./Ultra-face-mnn ./model/version-slim/slim-320.mnn ./model/version-slim/pfld-lite.mnn imgs/1.jpg
else
	./Ultra-face-mnn ./model/version-slim/slim-320.mnn ./model/version-slim/pfld-lite.mnn
fi
