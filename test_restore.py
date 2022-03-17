from torch.utils.data import IterableDataset, DataLoader, TensorDataset
from blur import *
from models import *
import argparse
import torchvision
import matplotlib.pyplot as plt

np.random.seed(2048)
torch.manual_seed(2048)

def test(model, data_loader):
    model = torch.load('./' + "restore_" + 'EMNIST' + "_" + str(args.data_type) + "_" + 'latest_model.pt')
    model.eval()
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, _ in data_loader:
            #images = random_rotate(images)
            inputs = torch.tensor(blur_filter(images), device=device, dtype=torch.float32)
            targets = images.to(device)
            output = model(inputs)
            loss = loss_func(output, targets)
            total += len(targets)
            total_loss += loss.item()

            id = 0
            for img in inputs:
                img = img.cpu().numpy()[0].astype(float)
                plt.imsave('./images/apply_filter/blur_' + args.data + "_" + str(id) + '.jpg', img, cmap='gray')
                id += 1

            id = 0
            for img in output:
                img = img.cpu().numpy()[0].astype(float)
                plt.imsave('./images/apply_filter/output_' + args.data + "_" + str(id) + '.jpg', img, cmap='gray')
                id += 1

            id = 0
            for img in targets:
                img = img.cpu().numpy()[0].astype(float)
                plt.imsave('./images/apply_filter/origin_' + args.data + "_" + str(id) + '.jpg', img, cmap='gray')
                id += 1

            break
    total_loss /= total
    print('Loss of the model on the Test images: %.8f' % total_loss)

    return total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--data', type=str, default='EMNIST')
    parser.add_argument('--data_type', type=str, default='original')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--patient', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--box_size', type=int, default=9)
    parser.add_argument('--kernel', type=int, default=9)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--isRestore', type=bool, default=False)
    args = parser.parse_args()

    print("Use GPU: " + str(torch.cuda.is_available()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = original_data(dataset=args.data)

    if args.data_type == 'move':
        blur_filter = move_blur(args.box_size, args.kernel, args.sigma)
    elif args.data_type == 'gaussian':
        blur_filter = torchvision.transforms.GaussianBlur(args.kernel, args.sigma)
    else:
        blur_filter = None

    random_rotate = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=180)
                ])

    test_loader = DataLoader(test_data, num_workers=1, batch_size=args.batch_size)

    inchannel, n, n_classes = 1, 0, 0
    if len(train_data.data[0].shape) == 2:
        n, _ = train_data.data[0].shape
    elif len(train_data.data[0].shape) == 3:
        inchannel, n, _ = train_data.data[0].shape
    try:
        n_classes = max(train_data.targets) + 1
    except:
        n_classes = max(train_data.labels) + 1

    model = RestoreCNN(inchannel, n)
    model.to(device)
    loss_func = nn.MSELoss()

    test(model, test_loader)