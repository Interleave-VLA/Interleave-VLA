### Libero 环境配置

配置环境时，安装 evdev 会报错，找不到很多 linux-header 中的宏定义，Conda 环境可能会导致编译时找不到系统头文件。
此时，可以尝试将 conda 环境中的编译器路径调整为系统默认路径：
    
```bash
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib"
```

robosuite v1.5.0 删除了 SingleArmEnv，因此无法使用单臂机械臂的可视化环境。
通过降级 robosuite 至 v1.4.0，可以使用 SingleArmEnv。

### TFDS build
经过 `tfds build` 转换过的数据格式必须要如下，不然就会读取报错。
```
dataset_name
├── 1.0.0 (合法的 version name)
│   ├── dataset_info.json
│   ├── features.json
│   ├── tfrecord instances...
```

tfds build 和 load 都会 post GCS 返回当前数据集书否存在于云端，



## 减小显存

动态图片选择或采样

## 上海学院项目位置

/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/yjc/cunxin/Chameleon-VLA