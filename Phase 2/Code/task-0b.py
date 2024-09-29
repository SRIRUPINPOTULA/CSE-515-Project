def main():
    print("Provide the Video File Name or VideoID: ")
    video_name = input()
    print(f"The video anem entered is: {video_name}")
    print("Enter the User selected Model: 1 - layer3 2 - layer4 3 - Avgpool, 4- BOG, 5-HISTOGRAM")
    feature_model = input()
    print(f"The feature model selected is: {feature_model}")
    print("The number of similar video enter value for m:")
    m = input()
    print(f"The number of similar video entered is: {m}")
    #Read the Files of target videos even and odd numbered
    #Measure
    #print the videos
main()