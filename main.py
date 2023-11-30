import torch 
from classify_model import LeNet5

def main(): 
    model = LeNet5() 
    model.load_state_dict(torch.load('model_1.pth'))
    model.eval()

    print(model)

if __name__ == "__main__": 
    main() 
