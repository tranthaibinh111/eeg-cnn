# motor impairment neural disorders
from .eeg_service import EEGService
from .colostate_service import ColostateService

from .simple_cnn_service import SimpleCNNService
from .lenet_service import LeNetService
from .alexnet_service import AlexNetService
from .vgg16_service import VGG16Service
from .cnn_service import CNNService

from .grega_vrbancic_service import GregaVrbancicService

# dependency injection
from .container import IocContainer
