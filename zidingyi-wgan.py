import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 加载数据函数
def load_data(img_dir, csv_file):
    df = pd.read_csv(csv_file)
    images = []
    conditions = []
    for idx, row in df.iterrows():
        img_path = os.path.join(img_dir, str(int(row[0])) + ".tif")
        img = load_img(img_path, color_mode='rgb')
        img = img.resize((256, 256))
        img = img_to_array(img)
        images.append(img)
        conditions.append(row[1:].values)
    return np.array(images), np.array(conditions)

# 数据路径
img_dir = r'D:\Python\PycharmProjects\pythonProject\gan\cgan\alloy7050-2'
csv_file = r'D:\Python\PycharmProjects\pythonProject\gan\cgan\alloy7050-2.csv'
images, conditions = load_data(img_dir, csv_file)

# 数据预处理
images = images / 127.5 - 1.0
conditions = conditions.astype(np.float32)

# 划分数据集
train_images, test_images, train_conditions, test_conditions = train_test_split(images, conditions, test_size=0.2, random_state=42)

def build_generator(noise_dim, condition_dim):
    # 噪声输入和条件输入
    noise_input = layers.Input(shape=(noise_dim,))
    condition_input = layers.Input(shape=(condition_dim,))
    condition_projected = layers.Dense(128)(condition_input)

    # 逐元素点积
    merged_input = layers.Multiply()([noise_input, condition_projected])

    # 全连接层
    x = layers.Dense(4 * 4 * 256)(merged_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((4, 4, 256))(x)

    # 第一层：反卷积
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)  # 反卷积
    x = layers.LeakyReLU(alpha=0.2)(x)
    print("1st shape", x.shape)

    # 第二层：反卷积
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)  # 反卷积
    x = layers.LeakyReLU(alpha=0.2)(x)
    print("2nd shape", x.shape)

    # 第三层：反卷积
    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)  # 反卷积
    x = layers.LeakyReLU(alpha=0.2)(x)
    print("3rd shape", x.shape)

    # 第四层：反卷积
    x = layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same')(x)  # 反卷积
    x = layers.LeakyReLU(alpha=0.2)(x)
    print("4th shape", x.shape)

    # 第五层：反卷积
    x = layers.Conv2DTranspose(8, kernel_size=4, strides=2, padding='same')(x)  # 反卷积
    x = layers.LeakyReLU(alpha=0.2)(x)
    print("5th shape", x.shape)

    # 输出层：反卷积
    output_img = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)  # 反卷积
    print("output_img shape:", output_img.shape)

    # 构建模型
    return Model([noise_input, condition_input], output_img)


# 定义判别器
def build_discriminator():
    img_input = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(8, kernel_size=4, strides=2, padding='same')(img_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(16, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(32, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    condition_input = layers.Input(shape=(2,))
    condition = layers.Dense(32 * 32 * 3)(condition_input)
    condition = layers.Reshape((32, 32, 3))(condition)
    combined_input = layers.Concatenate()([x, condition])
    print("Combined input shape:", combined_input.shape)
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(combined_input )
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)  # WGAN 使用线性输出
    return Model([img_input, condition_input], x)

# 定义WGAN模型
def build_wgan(generator, discriminator):
    noise_dim = generator.input[0].shape[1]
    condition_dim = generator.input[1].shape[1]
    noise_input = layers.Input(shape=(noise_dim,))
    condition_input = layers.Input(shape=(condition_dim,))
    generated_img = generator([noise_input, condition_input])
    validity = discriminator([generated_img, condition_input])
    wgan = Model([noise_input, condition_input], validity)
    return wgan

# 参数设置
noise_dim = 128
condition_dim = 2
clip_value = 0.04  # 判别器权重剪裁范围
n_critic = 10   # 每训练1次生成器，训练判别器的次数
epochs = 1500
batch_size = 4
sample_interval = 100

# 模型初始化
generator = build_generator(noise_dim, condition_dim)
discriminator = build_discriminator()
generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
# generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
wgan = build_wgan(generator, discriminator)

# 设置生成图片的保存路径
output_dir = r'D:\Python\PycharmProjects\pythonProject\gan\cgan\zidingyi-out-alloy7050-2'
os.makedirs(output_dir, exist_ok=True)

def save_generated_images(epoch, generated_images):
    """保存生成的图片"""
    for i in range(len(generated_images)):
        file_path = os.path.join(output_dir, f'generated_image_{epoch}_{i}.jpg')
        plt.imsave(file_path, (generated_images[i] * 127.5 + 127.5).astype(np.uint8))  # 将图片像素值还原为 [0, 255]

def plot_loss_curves(generator_losses, discriminator_losses):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, label='Generator Loss')
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(loss_plot_path)
    plt.show()

# 自定义训练过程
@tf.function
def train_step(real_images, real_conditions):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, noise_dim])
    fake_images = generator([noise, real_conditions])

    with tf.GradientTape() as disc_tape:
        real_validity = discriminator([real_images, real_conditions])
        fake_validity = discriminator([fake_images, real_conditions])
        disc_loss = -tf.reduce_mean(real_validity) + tf.reduce_mean(fake_validity)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    for var in discriminator.trainable_variables:
        var.assign(tf.clip_by_value(var, -clip_value, clip_value))

    with tf.GradientTape() as gen_tape:
        fake_images = generator([noise, real_conditions])
        fake_validity = discriminator([fake_images, real_conditions])
        gen_loss = -tf.reduce_mean(fake_validity)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return disc_loss, gen_loss

# 训练过程
generator_losses = []
discriminator_losses = []
for epoch in range(epochs):
    for _ in range(n_critic):
        idx = np.random.randint(0, train_images.shape[0], batch_size)
        real_images = train_images[idx]
        real_conditions = train_conditions[idx]
        d_loss, g_loss = train_step(real_images, real_conditions)

    generator_losses.append(g_loss.numpy())
    discriminator_losses.append(d_loss.numpy())

    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")

        # 保存生成图片
        noise = tf.random.normal([batch_size, noise_dim])
        generated_images = generator([noise, real_conditions], training=False).numpy()
        save_generated_images(epoch, generated_images)

# 绘制损失曲线
plot_loss_curves(generator_losses, discriminator_losses)
