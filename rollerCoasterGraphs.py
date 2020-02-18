from rollercoasterModel import rate_rollercoaster, rate_rollercoaster_from_list
import matplotlib.pyplot as plt

def generate_plot(attr_index, minimumValue=0.0, maximumValue=60.0, numOfPoints=500, other_attributes=[43,13,77,1972,3.03,-1.48,1.79,2.4,5,42,0], save=True):
    X = []
    Y = []
    current_x = minimumValue
    attributes = other_attributes
    step = (maximumValue - minimumValue) / numOfPoints
    for i in range(numOfPoints):
        X.append(current_x)
        attributes[attr_index] = current_x
        Y.append(rate_rollercoaster_from_list(attributes))
        current_x += step

    plt.plot(X, Y)
    if save:
        plt.savefig('attr' + str(attr_index) + '.png')
    else:
        plt.show()
    plt.clf()
    return X, Y


#the min and max have been chosen to show the intervals where changing this value makes a meaningful difference
generate_plot(0, minimumValue=25)  #max speed
generate_plot(1, minimumValue=9, maximumValue=21) #average_speed
generate_plot(2, minimumValue=35, maximumValue=120) #ride time
generate_plot(3, minimumValue=500, maximumValue=3500) #ride_length
generate_plot(4, minimumValue=2, maximumValue=3) #max_pos_gs
generate_plot(5, minimumValue=-2, maximumValue=1) #max_neg_gs
generate_plot(6, minimumValue=1, maximumValue=3.5) #max_lateral_gs
generate_plot(7, minimumValue=0, maximumValue=5)#total_air_time
generate_plot(8, minimumValue=2, maximumValue=15)#drops
generate_plot(9, minimumValue=10, maximumValue=100)#highest_drop_height
generate_plot(10, minimumValue=1, maximumValue=4)#inversions