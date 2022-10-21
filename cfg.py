
class ConfigData():
    # path
    img_folder_root = r'\\10.80.100.13\d\BackendAOI\Data\PassNg_Images\22_XML-282\1_chipping'
    output_root = r'D:\BackendAOI\Python\cnn\save_model\XML-282_chipping'
    output_name = r'classifier_weights'
    pretrained_weights = r''
    folderlist = r''
    
    # param for data
    valid_keep_ratio = 0.15
    img_newsize = 400
    device_memory_ratio = 0.99
    workers = 2
    epochs = 1000
    early_stop = 100
    batch_size = 16
    num_classes = 2
    over_sampling_thresh = 1000
    over_sampling_scale = 2
    

class HypScratch():
    lr = 0.001
    lrf = 0.2
    momentum = 0.925
    weight_decay = 0.0004
    warmup_epochs = 3
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1

