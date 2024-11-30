from Interface import *

df = pd.read_csv('train.csv')
x_train = df.drop(columns='label')
y_train = df['label']

def convert(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > 0:
                img[i][j] = 1
    return img
def convert_data(x_train):
    for i in tqdm(range(len(x_train)), desc='converting data'):
        xi = x_train.iloc[i, :]
        img = np.array(xi).reshape(28, 28)
        con = convert(img)
        x_train.iloc[i, :] = np.array(con).reshape(784, )
        
convert_data(x_train)


def main():
    window = PygameInterface()
    window.fit(x_train, y_train, lr=0.001, batch_size=64, epochs=10000)
    window.init()
    window.run()
main()