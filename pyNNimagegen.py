import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Загрузка датасета
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()

# Нормализация изображений
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32") / 255.0

# Определение генератора
generator = keras.Sequential(
    [
        layers.Dense(7 * 7 * 256, input_shape=(100,)),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'),
    ],
    name="generator",
)

# Определение дискриминатора
discriminator = keras.Sequential(
    [
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Определение функций потерь
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Определение оптимизаторов
generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)


# Определение функции тренировки
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Тренировка модели
EPOCHS = 100
BATCH_SIZE = 64

for epoch in range(EPOCHS):
    for batch in range(train_images.shape[0] // BATCH_SIZE):
        images = train_images[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
        train_step(images)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1} complete")


# Определение функции для генерации изображений по текстовому описанию
def generate_image(text_description):
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    return generated_image


# Пример использования функции для генерации изображения по текстовому описанию
image = generate_image("число 6 на чёрном фоне")
