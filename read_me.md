## MMVC LAB Project Templete

This project is a training templete based on TensorFlow and Slim. It is designed for rapid model development and iteration by providing flexible infrence and all popular based network and pretrained model. Currently, maintained by:   **Daitao Xing** (dx383@nyu.edu), **Mengwei Ren**(mengw@nyu.edu) and **Liang Niu**(liang.niu@nyu.edu) from MMVC Lab, NYU/NYU Abudhabi.

### Features

- [x] **Data Loader**  
	Focus on performance and flexiability of data pipeline. User can define their own pipeline using native python code without concerning the details of feeding data.  
	The way of loading data always becomes the bottleneck of GPU usages. The speed of loading data in this templete is much efficient than feed_dict way or queue way. It runs 1.2~5x faster than the equivalent code because the threads will prefetch the batches and trainers don't need to wait for queues. 
- [x] **Based Network and pretrained model**  
	The templete has included all based network used for classification task such as vgg, inception, resnet and so on. All those codes are from slim project so you can easily restore the pretrained model provided on slim website.  
- [x] **Multi-GPU** 
	Support training models using multi-gpus without wasting speed.  
- [x] **Restore checkpoints and ignore some variables**
- [x] **Flexible**  
	User can define their summaries, evauation and callback hooks.
- [ ] **Evaluation**
- [ ] **Models for sgementation task** 
	Only contains u_net, more is coming soon~

### This templete is under developing