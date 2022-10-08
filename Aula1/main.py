#!/usr/bin/python3

# -------------------------------------------------------------------------------
# Name:        main
# Purpose:     Typing Test
#
# Author:      Guilherme Mota
#
# Created:     31/10/2021
# Copyright:   (c) Guilherme 2021
# -------------------------------------------------------------------------------


# Imports
import argparse
import random
import readchar
import pprint
from time import time, ctime
from collections import namedtuple
from colorama import Fore, Back, Style

def typingtest(timer, max_value):
    # Variables
    inputs = []
    number_of_types = 0
    number_of_hits = 0
    type_hit_average_duration = 0
    type_miss_average_duration = 0
    total_duration = 0
    hit_durations, miss_durations = [], []

    # Namedtuple
    Input = namedtuple('Input', ['requested', 'received', 'duration'])

    # Ask user input to start
    print('Press any key to start...')
    if str(readchar.readkey()) == ' ':
        print(Fore.RED + Back.YELLOW + "Test canceled before ending!" + Style.RESET_ALL)
        exit(0)

    test_start = ctime(time())
    test_start_seconds = time()

    while True:
        # Generate random lowercase letter
        lettertotype = chr(random.randint(97, 122))
        print(Fore.BLACK + Back.WHITE + "Type letter " + Fore.BLUE + lettertotype + " " + Style.RESET_ALL)
        time_before_typing = time()

        # Read key input from user
        lettertyped = str(readchar.readkey())
        number_of_types += 1
        time_after_typing = time()
        type_duration = time_after_typing - time_before_typing

        if (lettertyped == ' '):
            print(Fore.RED + Back.YELLOW + "Test canceled before ending!" + Style.RESET_ALL)
            exit(0)
        elif (lettertyped == lettertotype):
            print(Fore.BLACK + Back.WHITE + "You typed letter " + Fore.GREEN + lettertyped + " " + Style.RESET_ALL)
            number_of_hits += 1
            hit_durations.append(type_duration)
        else:
            print(Fore.BLACK + Back.WHITE + "You typed letter " + Fore.RED + lettertyped + " " + Style.RESET_ALL)
            miss_durations.append(type_duration)

        inputinfo = Input(lettertotype, lettertyped, type_duration)
        inputs.append(inputinfo)

        if (timer):
            seconds = time() - test_start_seconds
            if (seconds >= max_value):
                break
        else:
            if (number_of_types == max_value):
                break

    test_duration = time() - test_start_seconds
    test_end = ctime(time())
    accuracy = number_of_hits / number_of_types

    type_average_duration = test_duration/number_of_types
    if number_of_hits > 0:
        type_hit_average_duration = sum(hit_durations)/number_of_hits
    if number_of_types != number_of_hits:
        type_miss_average_duration = sum(miss_durations)/(number_of_types - number_of_hits)

    # Test complited information
    print(Fore.RED + Back.YELLOW + "Test completed!" + Style.RESET_ALL)

    my_dict = {'Test Start': test_start, 'Test End': test_end, 'Test duration': test_duration,
                'Inputs': inputs, 'Number of Types': number_of_types, 'Number of Hits': number_of_hits,
                'Accuracy': accuracy, 'Type Average Duration': type_average_duration,
                'Type Hit Average Duration': type_hit_average_duration,
                'Type Miss Average Duration': type_miss_average_duration}

    # Print typing test information
    pprint.pprint(my_dict)


def main():
    # Define argparse inputs
    parser = argparse.ArgumentParser(description="Definition of test mode")
    parser.add_argument("-utm", type=int, help="Maximum number of seconds for Time Mode")
    parser.add_argument("-mv", type=int, help="Maximum value of inputs for Number of Inputs Mode")

    # Parse arguments
    args = vars(parser.parse_args())

    # Verify user input
    if (args['mv'] is None and args['utm'] is None):
        print(Fore.RED + Back.YELLOW + 'You have to select a mode before starting the typing test!' + Style.RESET_ALL)
    elif (args['mv'] is not None and args['utm'] is not None):
        print(Fore.RED + Back.YELLOW + 'You can not select two modes simultaneously!' + Style.RESET_ALL)
    elif (args['mv'] is not None and args['mv'] > 0):
        typingtest(False, args['mv'])
    elif (args['utm'] is not None and args['utm'] > 0):
        typingtest(True, args['utm'])
    else:
        print(Fore.RED + Back.YELLOW + 'The maximum number must be positive and greater than 0!' + Style.RESET_ALL)

if __name__ == '__main__':
    main()