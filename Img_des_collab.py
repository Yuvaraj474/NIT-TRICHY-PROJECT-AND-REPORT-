import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import random
import csv
import os

# Load the EuroSAT dataset
dataset, info = tfds.load('eurosat', with_info=True, as_supervised=True)
train_dataset = dataset['train']

# Class descriptions with multiple descriptions for each class

class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
class_descriptions = {
    0: [
        "Annual Crop: Fields that are cultivated and harvested annually.",
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
    ],
    1: [
        "Forest: Areas covered by trees, typically forming a dense canopy.",
        "Forest:Includes both deciduous and coniferous forests.",
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
    2: [
        "Herbaceous Vegetation: Areas covered with non-woody plants.",
        "Herbaceous Vegetation: Includes grasslands, meadows, and wetlands.",
        "Herbaceous Vegetation: Typically found in temperate and tropical regions.",
        "Herbaceous Vegetation: Provides habitat for insects and small mammals.",
        "Herbaceous Vegetation: Important for grazing and hay production.",
        "Herbaceous Vegetation: Can be seasonal, with variations in plant growth."
        "Herbaceous Vegetation: Often includes a variety of wildflowers and grasses.",
        "Herbaceous Vegetation: Plays a key role in soil erosion control and land stabilization.",
        "Herbaceous Vegetation: Supports pollinators like bees and butterflies.",
        "Herbaceous Vegetation: Can thrive in disturbed areas and recover quickly after disturbances.",
        "Herbaceous Vegetation: Contributes to the overall biodiversity and health of ecosystems."
    ],
    3: [
        "Highway: Major roads designed for fast traffic.",
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
    4: [
        "Industrial: Areas used for manufacturing or large-scale production.",
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
    5: [
        "Pasture: Grassland areas where animals are grazed.",
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
    6: [
        "Permanent Crop: Fields with long-term crops like vineyards and orchards.",
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
    7: [
        "Residential: Areas where people live, including houses and apartment buildings.",
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
    8: [
        "River: Large natural streams of water flowing in a channel.",
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
    9: [
        "Sea/Lake: Large bodies of salt or fresh water.",
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
    ]
}

# Function to get description for a given image and label
def get_image_description(label):
    description = class_descriptions[label.numpy()]
    selected_description = random.choice(description)
    return selected_description

# Function to get the class name for a given label
def get_class_name(label):
    class_names = {
        0: "Annual Crop",
        1: "Forest",
        2: "Herbaceous Vegetation",
        3: "Highway",
        4: "Industrial",
        5: "Pasture",
        6: "Permanent Crop",
        7: "Residential",
        8: "River",
        9: "Sea/Lake"
    }
    return class_names[label.numpy()]

# Function to save descriptions to a CSV file
def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Class", "Description"])
        for row in data:
            writer.writerow(row)

# Function to filter dataset by class
def filter_dataset_by_class(dataset, target_class):
    return dataset.filter(lambda image, label: tf.equal(label, target_class))

# Function to describe images of a specific class from the dataset and save to CSV
def describe_images_of_class(dataset, class_label, num_images, output_csv):
    descriptions = []
    filtered_dataset = filter_dataset_by_class(dataset, class_label)
    class_name = get_class_name(tf.constant(class_label))

    for i, (image, label) in enumerate(filtered_dataset.take(num_images)):
        description = get_image_description(label)
        image_name = f"classified_images/{class_name}/image_{i+1}.jpg"

        # Ensure directory exists
        os.makedirs(os.path.dirname(image_name), exist_ok=True)

        # Save the image
        plt.imsave(image_name, image.numpy())

        descriptions.append([image_name, class_name, description])

    save_to_csv(descriptions, output_csv)
    return descriptions

# Specify the class label and number of images to describe
class_label_to_describe =3   # For example, 1 for Forest
num_images_to_describe = 3000  # Specify the number of images you want to describe
output_csv_file =  f"image_descriptions_{class_names[class_label_to_describe].replace(' ', '_')}.csv"

# Describe images of the specified class and save to CSV
descriptions = describe_images_of_class(train_dataset, class_label_to_describe, num_images_to_describe, output_csv_file)
print(f"Descriptions saved to {output_csv_file}")

# Print descriptions
for i, description in enumerate(descriptions, 1):
    print(f"Image {i}: {description}")

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")
