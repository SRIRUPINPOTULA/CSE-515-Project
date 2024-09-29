def main():
    video_name = input("Provide the Video File Name or VideoID: ")
    # TODO: Should we map the VideoID to video name?
    print(f"The video name entered is: {video_name}")
    
    feature_space = input("Select a Feature Space from the following: 1 - Layer3, 2 - Layer4, 3 - AvgPool, 4- HOG, 5 - HOF, 6 - Color Histogram : ")
    # TODO: map the feature space number to feature space name. Or take input as feature space name
    print(f"The feature model selected is: {feature_space}")

    m = input("Enter the m number of similar videos needed: ")
    print(f"The number of similar video entered is: {m}")

    # Read the Files of target videos even and odd numbered
    # Measure
    # print the videos

main()