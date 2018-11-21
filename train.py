from model.initialization import initialization
from config import conf

m = initialization(conf, train=True)[0]

print("Training START")
m.fit()
print("Training COMPLETE")
