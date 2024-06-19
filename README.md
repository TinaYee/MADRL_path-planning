# MDRL
基于MADRL的多无人机路径规划开源仿真项目

文件夹结构
  algorithms：MADDPG和MATD3算法主体部分代码
  utils：智能体训练过程、神经网络结构、经验回访池逻辑、噪声设置等
  multiagent-particle-envs-master：MADRL算法运行使用的MPE多智能体粒子环境
  model：存储算法训练模型
  results：存储算法运行结果，包括图和数据
  assets：MPE文件夹中部分示例场景的运行结果
主函数列表
  main-1_r：优化目标只考虑数据速率
  main-1_e：同时优化数据速率和无人机能耗
  main-1_aoi_e：同时优化信息年龄和无人机能耗
开源项目信息
  单位：西安电子科技大学
  实验条件： Windows 11 64bit、Intel(R) Core(TM) i5-10400 CPU 和 Intel(R) UHD Graphics 630 (1024 MB)
  仿真平台及运行环境：PyCharm 2022.1.1(Community Edition)、Python 3.7
