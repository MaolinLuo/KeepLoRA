# https://github.com/Thunderbeee/ZSCL/blob/main/mtil/src/datasets/collections.py
import os
import re
import copy
from PIL import Image

import torch.utils.data
from torchvision import datasets


def underline_to_space(s):
    return s.replace("_", " ")


class ClassificationDataset:
    def __init__(
        self,
        train_transforms,
        test_transforms,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ):
        self.name = "classification_dataset"
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.location = location
        self.batch_size = batch_size
        if batch_size_eval is None:
            self.batch_size_eval = batch_size
        else:
            self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.append_dataset_name_to_template = append_dataset_name_to_template

        self.train_dataset = self.test_dataset = None
        self.train_loader = self.test_loader = None
        self.classnames = None
        self.templates = None

    def stats(self):
        L_train = len(self.train_dataset)
        L_test = len(self.test_dataset)
        N_class = len(self.classnames)
        return L_train, L_test, N_class

    @property
    def template(self):
        if self.append_dataset_name_to_template:
            return lambda x: self.templates[0](x)[:-1] + f", from dataset {self.name}]."
        return self.templates[0]

    def process_labels(self):
        self.classnames = [underline_to_space(x) for x in self.classnames]

    def split_dataset(self, dataset, train_transforms=None, test_transforms=None, ratio=0.8):
        dataset_2 = copy.deepcopy(dataset)
        train_size = int(ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        _, test_dataset = torch.utils.data.random_split(
            dataset_2,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        if train_transforms:
            train_dataset.dataset.transform = train_transforms
        if test_transforms:
            test_dataset.dataset.transform = test_transforms
        return train_dataset, test_dataset
    
    @property
    def class_to_idx(self):
        return {v: k for k, v in enumerate(self.classnames)}


class Aircraft(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "aircraft"
        self.train_dataset = datasets.FGVCAircraft(
            self.location, split="train", download=True, transform=self.train_transforms
        )
        self.test_dataset = datasets.FGVCAircraft(
            self.location, split="test", download=True, transform=self.test_transforms
        )
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of aircraft.",
            lambda c: f"a photo of the {c}, a type of aircraft.",
        ]

    # def process_labels(self):
    #     label = self.classnames
    #     for i in range(len(label)):
    #         if label[i].startswith("7"):
    #             label[i] = "Boeing " + label[i]
    #         elif label[i].startswith("An") or label[i].startswith("ATR"):
    #             pass
    #         elif label[i].startswith("A"):
    #             label[i] = "Airbus " + label[i]


class Caltech101(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "caltech101"
        dataset = datasets.Caltech101(
            self.location, download=True
        )
        self.classnames = dataset.categories

        train_dataset, test_dataset = self.split_dataset(dataset, self.train_transforms, self.test_transforms)
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.classnames = [
            "off-center face",
            "centered face",
            "leopard",
            "motorbike",
            "accordion",
            "airplane",
            "anchor",
            "ant",
            "barrel",
            "bass",
            "beaver",
            "binocular",
            "bonsai",
            "brain",
            "brontosaurus",
            "buddha",
            "butterfly",
            "camera",
            "cannon",
            "side of a car",
            "ceiling fan",
            "cellphone",
            "chair",
            "chandelier",
            "body of a cougar cat",
            "face of a cougar cat",
            "crab",
            "crayfish",
            "crocodile",
            "head of a  crocodile",
            "cup",
            "dalmatian",
            "dollar bill",
            "dolphin",
            "dragonfly",
            "electric guitar",
            "elephant",
            "emu",
            "euphonium",
            "ewer",
            "ferry",
            "flamingo",
            "head of a flamingo",
            "garfield",
            "gerenuk",
            "gramophone",
            "grand piano",
            "hawksbill",
            "headphone",
            "hedgehog",
            "helicopter",
            "ibis",
            "inline skate",
            "joshua tree",
            "kangaroo",
            "ketch",
            "lamp",
            "laptop",
            "llama",
            "lobster",
            "lotus",
            "mandolin",
            "mayfly",
            "menorah",
            "metronome",
            "minaret",
            "nautilus",
            "octopus",
            "okapi",
            "pagoda",
            "panda",
            "pigeon",
            "pizza",
            "platypus",
            "pyramid",
            "revolver",
            "rhino",
            "rooster",
            "saxophone",
            "schooner",
            "scissors",
            "scorpion",
            "sea horse",
            "snoopy (cartoon beagle)",
            "soccer ball",
            "stapler",
            "starfish",
            "stegosaurus",
            "stop sign",
            "strawberry",
            "sunflower",
            "tick",
            "trilobite",
            "umbrella",
            "watch",
            "water lilly",
            "wheelchair",
            "wild cat",
            "windsor chair",
            "wrench",
            "yin and yang symbol",
        ]
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a painting of a {c}.",
            lambda c: f"a plastic {c}.",
            lambda c: f"a sculpture of a {c}.",
            lambda c: f"a sketch of a {c}.",
            lambda c: f"a tattoo of a {c}.",
            lambda c: f"a toy {c}.",
            lambda c: f"a rendition of a {c}.",
            lambda c: f"a embroidered {c}.",
            lambda c: f"a cartoon {c}.",
            lambda c: f"a {c} in a video game.",
            lambda c: f"a plushie {c}.",
            lambda c: f"a origami {c}.",
            lambda c: f"art of a {c}.",
            lambda c: f"graffiti of a {c}.",
            lambda c: f"a drawing of a {c}.",
            lambda c: f"a doodle of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a painting of the {c}.",
            lambda c: f"the plastic {c}.",
            lambda c: f"a sculpture of the {c}.",
            lambda c: f"a sketch of the {c}.",
            lambda c: f"a tattoo of the {c}.",
            lambda c: f"the toy {c}.",
            lambda c: f"a rendition of the {c}.",
            lambda c: f"the embroidered {c}.",
            lambda c: f"the cartoon {c}.",
            lambda c: f"the {c} in a video game.",
            lambda c: f"the plushie {c}.",
            lambda c: f"the origami {c}.",
            lambda c: f"art of the {c}.",
            lambda c: f"graffiti of the {c}.",
            lambda c: f"a drawing of the {c}.",
            lambda c: f"a doodle of the {c}.",
        ]


class CIFAR100(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "cifar100"

        self.train_dataset = datasets.CIFAR100(
            self.location, download=True, train=True, transform=self.train_transforms
            )
        self.test_dataset = datasets.CIFAR100(
            self.location, download=True, train=False, transform=self.test_transforms
            )
        self.classnames = self.test_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c : f'a photo of a {c}.',
            lambda c : f'a blurry photo of a {c}.',
            lambda c : f'a black and white photo of a {c}.',
            lambda c : f'a low contrast photo of a {c}.',
            lambda c : f'a high contrast photo of a {c}.',
            lambda c : f'a bad photo of a {c}.',
            lambda c : f'a good photo of a {c}.',
            lambda c : f'a photo of a small {c}.',
            lambda c : f'a photo of a big {c}.',
            lambda c : f'a photo of the {c}.',
            lambda c : f'a blurry photo of the {c}.',
            lambda c : f'a black and white photo of the {c}.',
            lambda c : f'a low contrast photo of the {c}.',
            lambda c : f'a high contrast photo of the {c}.',
            lambda c : f'a bad photo of the {c}.',
            lambda c : f'a good photo of the {c}.',
            lambda c : f'a photo of the small {c}.',
            lambda c : f'a photo of the big {c}.',
            ]


class DTD(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "dtd"
        self.train_dataset = datasets.DTD(
            self.location, split="train", download=True, transform=self.train_transforms
        )
        self.test_dataset = datasets.DTD(
            self.location, split="test", download=True, transform=self.test_transforms
        )
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f'a photo of a {c} thing.',
            lambda c: f'a photo of a {c} texture.',
            lambda c: f'a photo of a {c} pattern.',
            lambda c: f'a photo of a {c} object.',
            lambda c: f'a photo of the {c} thing.',
            lambda c: f'a photo of the {c} texture.',
            lambda c: f'a photo of the {c} pattern.',
            lambda c: f'a photo of the {c} object.',
        ]


class EuroSAT(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "eurosat"
        dataset = datasets.EuroSAT(
            self.location, download=True
        )
        train_dataset, test_dataset = self.split_dataset(dataset, self.train_transforms, self.test_transforms)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.classnames = [
            "annual crop land",
            "forest",
            "brushland or shrubland",
            "highway or road",
            "industrial buildings or commercial buildings",
            "pasture land",
            "permanent crop land",
            "residential buildings or homes or apartments",
            "river",
            "lake or sea",
        ]
        self.process_labels()
        self.templates = [
            lambda c: f"a centered satellite photo of {c}.",
            lambda c: f"a centered satellite photo of a {c}.",
            lambda c: f"a centered satellite photo of the {c}.",
        ]


class Flowers(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "flowers"
        self.train_dataset = datasets.Flowers102(
            self.location, split="train", download=True, transform=self.train_transforms
        )
        self.test_dataset = datasets.Flowers102(
            self.location, split="test", download=True, transform=self.test_transforms
        )
        self.classnames = [
            "pink primrose",
            "hard-leaved pocket orchid",
            "canterbury bells",
            "sweet pea",
            "english marigold",
            "tiger lily",
            "moon orchid",
            "bird of paradise",
            "monkshood",
            "globe thistle",
            "snapdragon",
            "colt's foot",
            "king protea",
            "spear thistle",
            "yellow iris",
            "globe-flower",
            "purple coneflower",
            "peruvian lily",
            "balloon flower",
            "giant white arum lily",
            "fire lily",
            "pincushion flower",
            "fritillary",
            "red ginger",
            "grape hyacinth",
            "corn poppy",
            "prince of wales feathers",
            "stemless gentian",
            "artichoke",
            "sweet william",
            "carnation",
            "garden phlox",
            "love in the mist",
            "mexican aster",
            "alpine sea holly",
            "ruby-lipped cattleya",
            "cape flower",
            "great masterwort",
            "siam tulip",
            "lenten rose",
            "barbeton daisy",
            "daffodil",
            "sword lily",
            "poinsettia",
            "bolero deep blue",
            "wallflower",
            "marigold",
            "buttercup",
            "oxeye daisy",
            "common dandelion",
            "petunia",
            "wild pansy",
            "primula",
            "sunflower",
            "pelargonium",
            "bishop of llandaff",
            "gaura",
            "geranium",
            "orange dahlia",
            "pink-yellow dahlia",
            "cautleya spicata",
            "japanese anemone",
            "black-eyed susan",
            "silverbush",
            "californian poppy",
            "osteospermum",
            "spring crocus",
            "bearded iris",
            "windflower",
            "tree poppy",
            "gazania",
            "azalea",
            "water lily",
            "rose",
            "thorn apple",
            "morning glory",
            "passion flower",
            "lotus",
            "toad lily",
            "anthurium",
            "frangipani",
            "clematis",
            "hibiscus",
            "columbine",
            "desert-rose",
            "tree mallow",
            "magnolia",
            "cyclamen",
            "watercress",
            "canna lily",
            "hippeastrum",
            "bee balm",
            "ball moss",
            "foxglove",
            "bougainvillea",
            "camellia",
            "mallow",
            "mexican petunia",
            "bromelia",
            "blanket flower",
            "trumpet creeper",
            "blackberry lily",
        ]
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of flower.",
        ]


class Food(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "food"
        self.train_dataset = datasets.Food101(
            self.location, split="train", download=True, transform=self.train_transforms
        )
        self.test_dataset = datasets.Food101(
            self.location, split="test", download=True, transform=self.test_transforms
        )
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of food.",
        ]


class MNIST(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "mnist"
        self.train_dataset = datasets.MNIST(
            self.location, train=True, download=True, transform=self.train_transforms
        )
        self.test_dataset = datasets.MNIST(
            self.location, train=False, download=True, transform=self.test_transforms
        )
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f'a photo of the number: "{c}".',
        ]


class OxfordPet(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "oxford pet"
        self.train_dataset = datasets.OxfordIIITPet(
            self.location, split="trainval", download=True, transform=self.train_transforms
        )
        self.test_dataset = datasets.OxfordIIITPet(
            self.location, split="test", download=True, transform=self.test_transforms
        )
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}, a type of pet.",
        ]


class StanfordCars(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "stanford cars"
        self.train_dataset = datasets.StanfordCars(
            self.location, split="train", download=False, transform=self.train_transforms
        )
        self.test_dataset = datasets.StanfordCars(
            self.location, split="test", download=False, transform=self.test_transforms
        )
        self.classnames = self.train_dataset.classes
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of the {c}, a type of car.",
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
            lambda c: f"a photo of my {c}.",
            lambda c: f"i love my {c}!",
            lambda c: f"a photo of my dirty {c}.",
            lambda c: f"a photo of my clean {c}.",
            lambda c: f"a photo of my new {c}.",
            lambda c: f"a photo of my old {c}.",
        ]


class SUN397(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "sun397"
        dataset = datasets.SUN397(
            self.location, download=True
        )
        train_dataset, test_dataset = self.split_dataset(dataset, self.train_transforms, self.test_transforms)
        self.classnames = dataset.classes

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of the {c}.",
            lambda c: f"a photo of a {c}.",
        ]


class Places(ClassificationDataset, torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "places"
        
        # Load classnames from classnames.txt
        classnames_path = os.path.join(self.location, "classnames.txt")
        with open(classnames_path, 'r') as f:
            self.classnames = []
            for line in f:
                line = line.strip()
                parts = line.split(' ', 1)
                self.classnames.append(parts[1])
        
        # No training dataset for Places
        self.train_dataset = None
        
        # Load test samples directly
        annotation_file = os.path.join(self.location, "Places_test.txt")
        self.samples = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split(' ')
                img_path = parts[0]
                label = int(parts[1])
                self.samples.append((img_path, label))
        
        # Set this instance as the test dataset
        self.test_dataset = self
        
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}."
        ]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.location, img_path)
        
        image = Image.open(full_path).convert('RGB')
        
        if self.test_transforms:
            image = self.test_transforms(image)
        
        return image, label

    def stats(self):
        """Override stats method since there's no training dataset."""
        L_train = 0
        L_test = len(self.samples)
        N_class = len(self.classnames)
        return L_train, L_test, N_class


class ImageNet(ClassificationDataset, torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "imagenet"
        
        # Load classnames from classnames.txt
        classnames_path = os.path.join(self.location, "classnames.txt")
        with open(classnames_path, 'r') as f:
            self.classnames = []
            for line in f:
                line = line.strip()
                parts = line.split(' ', 1)
                self.classnames.append(parts[1])
        
        # No training dataset for ImageNet
        self.train_dataset = None
        
        # Load test samples directly
        annotation_file = os.path.join(self.location, "ImageNet_test.txt")
        self.samples = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split(' ')
                img_path = parts[0]
                label = int(parts[1])
                self.samples.append((img_path, label))
        
        # Set this instance as the test dataset
        self.test_dataset = self
        
        self.process_labels()
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.location, img_path)
        
        image = Image.open(full_path).convert('RGB')
        
        if self.test_transforms:
            image = self.test_transforms(image)
        
        return image, label

    def stats(self):
        """Override stats method since there's no training dataset."""
        L_train = 0
        L_test = len(self.samples)
        N_class = len(self.classnames)
        return L_train, L_test, N_class

