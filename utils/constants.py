"""Constants used in the code."""

dataset_name_acronym = {
    "caltech101": "CAL101",
    "cifar100": "CI100",
    "country211": "CO211",
    "cub200": "CUB200",
    "dtd": "DTD",
    "eurosat": "EUSAT",
    "fgvc_aircraft": "AirCr.",
    "food101": "Food101",
    "gtsrb": "GTSRB",
    "mini_imagenet": "MiniIN",
    "oxford_flowers": "FLO102",
    "oxford_pets": "Pets",
    "resisc45": "RES45",
    "stanford_cars": "Cars",
    "sun397": "SUN397",
    "voc2007": "VOC",
}

encoder_name_acronym = {
    "google/vit-base-patch16-224": "ViT-B16-224",
    "google/vit-base-patch16-224-in21k": "ViT-B16-224-21k",
    "google/vit-base-patch16-384": "ViT-B16-384",
    "google/vit-base-patch32-224-in21k": "ViT-B32-224-21k",
    "google/vit-base-patch32-384": "ViT-B32-384",
    "google/vit-large-patch16-224": "ViT-L16-224",
    "google/vit-large-patch16-224-in21k": "ViT-L16-224-21k",
    "google/vit-large-patch16-384": "ViT-L16-384",
    "google/vit-large-patch32-224-in21k": "ViT-L32-224-21k",
    "google/vit-large-patch32-384": "ViT-L32-384",
    "google/vit-huge-patch14-224-in21k": "ViT-H14-224-21k",
}


dataset_roots = {
    "cub200": "data/CUB_200_2011",
    "mini_imagenet": "data/miniimagenet",
}

fscil_base_classes = {
    "cifar100": 60,
    "cub200": 100,
    "mini_imagenet": 60,
}

fscil_ways = {
    "cifar100": 5,
    "cub200": 10,
    "mini_imagenet": 5,
}

num_classes = {
    "caltech101": 102,
    "cifar100": 100,
    "country211": 211,
    "cub200": 200,
    "dtd": 47,
    "eurosat": 10,
    "fgvc_aircraft": 100,
    "food101": 101,
    "gtsrb": 43,
    "mini_imagenet": 100,
    "oxford_flowers": 102,
    "oxford_pets": 37,
    "resisc45": 45,
    "stanford_cars": 196,
    "sun397": 397,
    "voc2007": 20,
}

fscit_base_classes = {
    "sun397": 40,
    "dtd": 5,
    "voc2007": 2,
    "stanford_cars": 20,
    "resisc45": 5,
    "oxford_pets": 5,
    "oxford_flowers": 12,
    "gtsrb": 7,
    "fgvc_aircraft": 10,
    "eurosat": 1,
    "country211": 22,
    "caltech101": 12,
    "food101": 11,
    "cifar100": 10,
    "cub200": 20,
    "mini_imagenet": 10,
}

fscit_ways = {
    "sun397": 40,
    "dtd": 5,
    "voc2007": 2,
    "stanford_cars": 20,
    "resisc45": 5,
    "oxford_pets": 4,
    "oxford_flowers": 10,
    "gtsrb": 4,
    "fgvc_aircraft": 10,
    "eurosat": 1,
    "country211": 21,
    "caltech101": 10,
    "food101": 10,
    "cifar100": 10,
    "cub200": 20,
    "mini_imagenet": 20,
}
