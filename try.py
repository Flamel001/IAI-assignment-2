import numpy as np
import random as rand
from PIL import Image, ImageDraw
from scipy.spatial import distance
from IPython.display import display

#the only not random function:)
#fitness function returns distance because that is used as weight for priority array
#Using this fitness function I'm trying to make my "ART PICTURE" MAX FAR FROM "INITIAL PICTURE"
#So, higher distance -> higher priority to be chosen as parent
def fitness(vector):
    return distance.euclidean(vector, initial_vector)


#parents are chosen randomly. higher the priority(fitness function) ->higher the chance
#to be chosen as parent. Parents are always different
def parents_choice(population,priority):
    parent1 = rand.choices(population,weights=priority)
    parent1 = np.array(parent1)
    parent2 = rand.choices(population,weights=priority)
    parent2 = np.array(parent2)
    while(np.array_equal(parent1,parent2)):
        parent2 = rand.choices(population,weights=priority)
        parent2 = np.array(parent2)
    return parent1, parent2


#every two parents "return" two offsprings
def crossover(parent1,parent2):
    #changer stands for adding random:)
    changer = np.random.randint(low=0,high=2)
    #randomly chosen amount of data taken from the first parent
    crossover1 = np.random.randint(low = 1, high = 786432)
    crossover2 = 786432-crossover1
    if changer!=1:
        offspring1 =np.concatenate((parent1[0][0:crossover1],parent2[0][crossover1:]), axis=None)
        offspring2 =np.concatenate((parent1[0][0:crossover2],parent2[0][crossover2:]), axis=None)

    else:
        offspring1 =np.concatenate((parent2[0][0:crossover1],parent1[0][crossover1:]), axis=None)
        offspring2 =np.concatenate((parent2[0][0:crossover2],parent1[0][crossover2:]), axis=None)
    offspring1 = np.array(offspring1)
    return offspring1, offspring2



#addition of random to random random:)
#vector -> picture -> Add mutation(ellipse of random color) to picture ->back to vector
def mutation(offspring):
    matrix = offspring.reshape((512, 512, 3))
    img = Image.fromarray(np.uint8(matrix))
    draw = ImageDraw.Draw(img)
    r = np.random.randint(low=0,high=256)
    g = np.random.randint(low=0,high=256)
    b = np.random.randint(low=0,high=256)
    x1 = np.random.randint(low=0,high=512)
    y1 = np.random.randint(low=0,high=512)
    x_offset = np.random.randint(low=5,high=30)
    y_offset = np.random.randint(low=5,high=30)
    changer = np.random.randint(low=0,high=2)
    if changer<1:
        x2=x1+x_offset
        y2=y1+y_offset
    else:
        x2=x1-x_offset
        y2=y1-y_offset
        x1,x2 = x2,x1
        y1,y2 = y2,y1
    draw.ellipse((x1,y1,x2,y2), fill=(r,g,b))
    img = np.array(img)
    return img.reshape(img.shape[0]*img.shape[1]*img.shape[2])


img = Image.open('pleshivii.jpg')
img = img.resize((512,512))
img.show()
img = np.array(img)
print("img shape is" + str(img.shape))
#we will always compare with initial vector to calculate fitness function
initial_vector = img.reshape(img.shape[0]*img.shape[1]*img.shape[2])

#initial
red_image = np.array(Image.new('RGB', (512,512), 'red'))
red_vector = red_image.reshape(red_image.shape[0]*red_image.shape[1]*red_image.shape[2])  

blue_image = np.array(Image.new('RGB', (512,512), 'blue'))
blue_vector = blue_image.reshape(blue_image.shape[0]*blue_image.shape[1]*blue_image.shape[2])

green_image = np.array(Image.new('RGB', (512,512), 'green'))
green_vector = green_image.reshape(green_image.shape[0]*green_image.shape[1]*green_image.shape[2])

white_image = np.array(Image.new('RGB', (512,512), 'white'))
white_vector = white_image.reshape(white_image.shape[0]*white_image.shape[1]*white_image.shape[2])

black_image = np.array(Image.new('RGB', (512,512), 'black'))
black_vector = black_image.reshape(black_image.shape[0]*black_image.shape[1]*black_image.shape[2])

#global arrays to store population and their priorities
population_arr = [red_vector,green_vector,blue_vector,white_vector,black_vector]
priority_arr = []
for i in population_arr:
    priority_arr.append(fitness(i))
print(priority_arr)

for i in range(50001):
    parent1, parent2 = parents_choice(population_arr,priority_arr)
    offspring1, offspring2 = crossover(parent1,parent2)
    mutated_offspring1 = mutation(offspring1)
    mutated_offspring2 = mutation(offspring2)
    population_arr.append(mutated_offspring1)
    priority_arr.append(fitness(mutated_offspring1))
    population_arr.append(mutated_offspring2)
    priority_arr.append(fitness(mutated_offspring2))
    if len(population_arr)>50:
        for j in range(40):
            min_elem_index = priority_arr.index(min(priority_arr))
            priority_arr.pop(min_elem_index)
            population_arr.pop(min_elem_index)
    #if (i % 1000 ==0):
max_elem_index = priority_arr.index(max(priority_arr))
my_matrix = population_arr[max_elem_index].reshape((512, 512, 3))
img = Image.fromarray(np.uint8(my_matrix))
print('Generation number is ' + str(i))
img.show()
