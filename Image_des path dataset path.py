import tensorflow as tf
import os
import random
import csv

# Define constants and paths
TRAIN_DATASET_PATH = r'C:\Users\yuvar\Desktop\Image_description\CTrain'
VAL_DATASET_PATH = r'C:\Users\yuvar\Desktop\Image_description\CVal'
IMG_SIZE = (224, 224)  # Adjust based on your model's input size

# Class names and their corresponding descriptions
class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
class_descriptions = {
    0: ["Annual Crop: Fields that are cultivated and harvested annually.",
        "Annual Crop: Typical examples include wheat, corn, and barley.",
        "Annual Crop: These crops require annual planting and harvesting.",
        "Annual Crop: Fields often have a regular, grid-like pattern.",
        "Annual Crop: Can vary significantly in appearance throughout the year.",
        "Annual Crop: Often found in agricultural regions."
        "Annual Crop:  Requires specific growing conditions and careful management.",
        "Annual Crop: Includes vegetables like carrots, tomatoes, and peppers.",
        "Annual Crop: Contributes significantly to food production and economy.",
        "Annual Crop: Often grown in rotation to maintain soil health.",
        "Annual Crop:  Provides raw materials for various industries.",
        ],  # Add more descriptions
    1: ["Forest: Areas covered by trees, typically forming a dense canopy.",
        "Forest: Includes both deciduous and coniferous forests.",
        "Forest:Provides habitat for a wide range of wildlife.",
        "Forest:Characterized by high biomass and canopy cover.",
        "Forest:Can be found in mountainous and lowland regions.",
        "Forest:Important for carbon sequestration."
        "Forest: Plays a crucial role in regulating the Earth's climate.",
        "Forest: Serves as a source of timber and other forest products.",
        "Forest:Includes diverse ecosystems such as rainforests, boreal forests, and temperate forests.",
        "Forest: Supports numerous recreational activities like hiking, camping, and bird-watching.",
        "Forest: Essential for maintaining biodiversity and ecological balance.",
        ],
    2: ["HerbaceousVegetation: Areas covered by non-woody plants.",
        "HerbaceousVegetation: Includes grasslands , meadows and wetlands.",
        "HerbaceousVegetation: Typically found in temperate and tropical regions.",
        "HerbaceousVegetation: Provides habitat for insects and small mammals.",
        "HerbaceousVegetation: Important for grazing and hay production.",
        "HerbaceousVegetation: Can be seasonal, with variations in plant growth."
        "HerbaceousVegetation: Often includes a variety of wildflowers and grasses.",
        "HerbaceousVegetation: Plays a key role in soil erosion control and land stabilization.",
        "HerbaceousVegetation: Supports pollinators like bees and butterflies.",
        "HerbaceousVegetation: Can thrive in disturbed areas and recover quickly after disturbances.",
        "HerbaceousVegetation: Contributes to the overall biodiversity and health of ecosystems."
        ],

    3: ["Highway: Major roads designed for fast traffic.",
        "Highway: Characterized by multiple lanes and high speed limits.",
        "Highway: Includes motorways and expressways.",
        "Highway: Often lined with barriers and signs.",
        "Highway: Important for regional and national transportation.",
        "Highway: Can be found connecting major cities and regions."
        "Highway: Equipped with rest areas and service stations for travelers.",
        "Highway: Features interchanges and overpasses for efficient traffic flow.",
        "Highway: Constructed with durable materials to withstand heavy usage.",
        "Highway: Monitored by traffic cameras and patrolled by law enforcement.",
        "Highway:  Can include toll roads to manage funding and maintenance costs."
        ],
    4: ["Industrial: Areas used for manufacturing or large-scale production.",
        "Industrial: Includes factories, warehouses, and plants.",
        "Industrial: Characterized by large buildings and machinery.",
        "Industrial: Often located near transportation hubs.",
        "Industrial: Important for economic activity and employment.",
        "Industrial: Can be sources of pollution and environmental impact."
        "Industrial: Requires significant infrastructure such as power and water supply.",
        "Industrial: Zoned specifically for commercial and industrial use.",
        "Industrial: Can operate 24/7 to meet production demands.",
        "Industrial: Often has strict safety and regulatory standards",
        "Industrial: Plays a crucial role in the supply chain and logistics."
         ],
    5: ["Pasture: Grassland areas where animals are grazed.",
        "Pasture:Includes fields for cattle, sheep, and horses.",
        "Pasture:Characterized by open, grassy areas.",
        "Pasture:Important for livestock production.",
        "Pasture:Can be rotationally grazed to maintain grass health.",
        "Pasture:Often found in rural and agricultural regions.",
        "Pasture:Requires regular maintenance and management.",
        "Pasture:Provides natural feed for grazing animals.",
        "Pasture:Can include a mix of grasses and legumes for better nutrition.",
        "Pasture:Often integrated into sustainable farming practices.",
        "Pasture:May be supplemented with hay or silage during off-seasons."
],
    6: ["Permanent Crop: Fields with long-term crops like vineyards and orchards.",
        "Permanent Crop:Includes fruit trees, nuts, and grapevines.",
        "Permanent Crop:Characterized by rows of perennial plants.",
        "Permanent Crop:Important for producing high-value crops.",
        "Permanent Crop:Require long-term maintenance and care.",
        "Permanent Crop:Often found in temperate and Mediterranean climates."
        "Permanent Crop:Often supported by irrigation systems for consistent water supply.",
        "Permanent Crop:Includes trees like olive, almond, and citrus.",
        "Permanent Crop:Provides habitat for beneficial insects and birds.",
        "Permanent Crop:Can be managed organically to minimize chemical inputs.",
        "Permanent Crop:Requires specialized pruning and training techniques."
         ],
    7: ["Residential: Areas where people live, including houses and apartment buildings.",
        "Residential:Includes suburbs, urban neighborhoods, and rural homes.",
        "Residential:Characterized by buildings, roads, and green spaces.",
        "Residential:Important for providing housing and community spaces.",
        "Residential:Can vary significantly in density and layout.",
        "Residential:Often includes amenities like parks and schools.",
        "Residential:Includes diverse architectural styles and housing types.",
        "Residential:Connected by local amenities like shopping centers and healthcare facilities.",
        "Residential:Often designed with pedestrian-friendly pathways and bike lanes.",
        "Residential:Features sustainable practices like rainwater harvesting and solar panels.",
        "Residential: Supported by neighborhood associations for community engagement and improvement projects."
        ],
    8: ["River: Large natural streams of water flowing in a channel.",
        "River: Includes major rivers and their tributaries.",
        "River: Characterized by flowing water and riparian zones.",
        "River: Important for water supply, transportation, and ecosystems.",
        "River: Can vary in width, depth, and flow rate.",
        "River: Often found in valleys and lowland areas."
        "River: Supports diverse aquatic habitats and species.",
        "River: Shapes the landscape through erosion and sediment deposition.",
        "River: Provides recreational opportunities like boating, fishing, and rafting.",
        "River: Can be harnessed for hydroelectric power generation.",
        "River: Vulnerable to pollution and habitat degradation from human activities."
        ],
    9: ["Sea/Lake: Large bodies of salt or fresh water.",
        "Sea/Lake: Includes oceans, seas, and large freshwater lakes.",
        "Sea/Lake: Characterized by open water and shorelines.",
        "Sea/Lake: Important for biodiversity, recreation, and resources.",
        "Sea/Lake: Can vary in size, depth, and water quality.",
        "Sea/Lake: Often found in coastal and inland regions.",
        "Sea/Lake: Sustains commercial fisheries and aquaculture industries.",
        "Sea/Lake: Influences local climate and weather patterns.",
        "Sea/Lake: Provides habitats for migratory birds and waterfowl.",
        "Sea/Lake: Subject to environmental challenges like eutrophication and habitat loss.",
        "Sea/Lake: Supports cultural traditions and indigenous communities."
        ],# Add more descriptions
    # Add descriptions for other classes
}

class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}

# Function to preprocess image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to load and preprocess image
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Function to get dataset from path
def get_dataset_from_path(dataset_path):
    all_image_paths = []
    all_labels = []

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            all_image_paths.append(img_path)
            all_labels.append(class_to_label[class_name])

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(all_labels)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds

# Function to get image description
def get_image_description(label):
    descriptions = class_descriptions[label.numpy()]
    selected_description = random.choice(descriptions)
    return selected_description

# Function to save descriptions to a CSV file
def save_descriptions_to_csv(dataset, class_label, output_csv):
    descriptions = []

    class_name = class_names[class_label]
    filtered_dataset = dataset.filter(lambda image, label: tf.equal(label, class_label))

    for i, (image, label) in enumerate(filtered_dataset):
        description = get_image_description(label)
        image_name = f"classified_images/{class_name}/image_{i+1}.jpg"

        # Ensure directory exists
        os.makedirs(os.path.dirname(image_name), exist_ok=True)

        # Save the image
        #plt.imsave(image_name, image.numpy())

        descriptions.append([image_name, class_name, description])

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Class", "Description"])
        for row in descriptions:
            writer.writerow(row)

    return descriptions

# Load the training and validation datasets
train_dataset = get_dataset_from_path(TRAIN_DATASET_PATH)
val_dataset = get_dataset_from_path(VAL_DATASET_PATH)

# Specify the class label to describe
class_label_to_describe = 2  # For example, 2 for Herbaceous Vegetation
output_csv_file = f"image_descriptions_{class_names[class_label_to_describe].replace(' ', '_')}.csv"

# Describe images of the specified class and save to CSV
descriptions = save_descriptions_to_csv(train_dataset, class_label_to_describe, output_csv_file)
print(f"Descriptions saved to {output_csv_file}")

# Print descriptions
for i, description in enumerate(descriptions, 1):
    print(f"Image {i}: {description}")

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")
