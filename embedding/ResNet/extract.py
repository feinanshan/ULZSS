import torchvision.models as models
import pickle as pkl

resnet101 = models.resnet101(pretrained=True)
fc_layer = resnet101.fc.weight.detach().numpy()

print(fc_layer.shape)

with open('res101_fc.pkl', 'wb') as fp:
    pkl.dump(fc_layer, fp)


#pickle.load(open("res101_fc.pkl", 'rb'))
