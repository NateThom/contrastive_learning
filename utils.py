import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path',
                        default='/home/nthom/Documents/datasets/CelebA/Img/',
                        # default='/home/nthom/Documents/datasets/CelebA/Img/partial_blackout/',
                        #default='/home/nthom/Documents/datasets/lfwa/',
                        # default='/home/nthom/Documents/datasets/UMD-AED/',
                        help='Path to input data directory [/home/user/Documents/input_images/]')

    parser.add_argument('--image_dir',
                        default='resized_images_178x218/',
                        # default='img_celeba/',
                        # default='resized_aligned_images_178x218',
                        # default='resized_segment1',
                        # default='lfw',
                        # default='croppedImages',
                        help='input_images')

    parser.add_argument('--attr_label_path',
                        # default='/home/nthom/Documents/datasets/UNR_Facial_Attribute_Parsing_Dataset/list_attr_celeba_attparsenet.csv',
                        default='/home/nthom/Documents/datasets/UNR_Facial_Attribute_Parsing_Dataset/list_attr_celeba_hair.csv',
                        # default='/home/nthom/Documents/datasets/lfwa/lfwa_labels_full_paths.csv',
                        # default='/home/nthom/Documents/datasets/UMD-AED/Files_attparsenet/list_attr_umdaed_reordered.csv',
                        help='Path to mapping between input images and binary attribute labels [/home/user/Documents/list_attr_celeba_attparsenet.csv]')

    parser.add_argument('--load',
                        default=False,
                        # default=True,
                        help='True for loading a pretrained model, False otherwise [0]')

    parser.add_argument('--load_path',
                        default='/home/nthom/Documents/contrastive_learning/checkpoints/',
                        help='File path for the model to load [/home/user/Document/models/]')

    parser.add_argument('--load_file',
                        default='epoch=24-Validation Loss=0.24166-resnet18Pretrain_resizedImages178x218_hair_randomResizedCrop_blur15_hFlip_0.01.ckpt',
                        help='File name for the model to load [/model_to_load]')

    parser.add_argument('--save',
                        default=True,
                        # default=False,
                        help='True for saving the model, False otherwise [True]')

    parser.add_argument('--save_path',
                        default='/home/nthom/Documents/contrastive_learning/checkpoints',
                        help='Dir for saving models [./saved_models/]')

    # Base Model, Dataset, labels, data augs, Learning Rate
    parser.add_argument('--save_name',
                        default='resnet152Pretrain_resizedImages178x218_hair_randomResizedCrop_0.01',
                        help='Dir for saving models [./saved_models/]')

    parser.add_argument('--train_epochs',
                        default=25,
                        help='Number of training epochs [22]')

    parser.add_argument('--train_size',
                        #lfwa and umd
                        # default=0,
                        #celeba
                        default=162771,
                        help='Number of samples in training set [162770]')

    parser.add_argument('--val_size',
                        #lfwa and umd
                        # default=0,
                        #celaba
                        default=19867,
                        help='Number of samples in validation set [19867]')

    parser.add_argument('--test_size',
                        #lfwa
                        # default=13088,
                        # umd
                        # default=2808,
                        #celeba
                        default=19961,
                        help='Number of samples in test set [19963]')

    parser.add_argument('--all_size',
                        #lfwa
                        # default=13088,
                        # umd
                        # default=2808,
                        #celeba
                        default=202599,
                        help='Total Number of samples in the dataset [202600]')

    parser.add_argument('--train',
                        # default=False,
                        default=True,
                        help='Train the model on the training set and evaluate on the validation set')

    parser.add_argument('--val_only',
                        default=False,
                        # default=True,
                        help='Evaluate the model on the validation set')

    parser.add_argument('--test',
                        default=False,
                        # default=True,
                        help='Evaluate the model on the test set')

    parser.add_argument('--pretrain',
                        # default=False,
                        default=True,
                        help='Download pretrained resnet weights')

    parser.add_argument('--n_labels',
                        default=5,
                        # default=40,
                        help='Number of classes in task')

    parser.add_argument('--attr_to_use',
                        # default=['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
                        default=['Other', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
                        # default=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                        #          'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                        #          'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                        #          'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                        #          'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                        #          'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                        #          'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                        #          'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'],
                        help='List of attributes to predict')

    parser.add_argument('--attr_list',
                        default=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                                'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                                'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                                'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                                'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                                'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                                'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'],
                        help='List of all 40 attributes')

    parser.add_argument('--show_batch',
                        default=False,
                        # default=True,
                        help='Show the batch input images and masks for debugging')

    parser.add_argument('--shuffle',
                        # default=False,
                        default=True,
                        help='Shuffle the order of training samples. Validation and Testing sets will not be shuffled [True]')

    parser.add_argument('--random_seed',
                        default=256,
                        help='Seed for random number generators [64]')

    parser.add_argument('--batch_size',
                        default=64,
                        help='Batch size for images [32]')

    parser.add_argument('--lr',
                        default=0.01,
                        help='Learning rate [0.001]')

    parser.add_argument('--patience',
                        default=5,
                        help='Learning Rate Scheduler Patience [5]')

    return parser.parse_args()

    # parser.add_argument('--model',
    #                     default="attparsenet",
    #                     help='Designates the model to be initialized [attparsenet]')
    # parser.add_argument('--save_feature_maps',
    #                     default=False,
    #                     # default=True,
    #                     help='Save all feature maps for data in either the test or val set')
