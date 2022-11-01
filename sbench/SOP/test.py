# Make an app that randomly lottery generates lottery numbers, that have 6 digits
# and then lets the user guess the lottery number

import random

def lottery():
    return random.randint(000000, 999999)

def main():
    lottery_number = lottery()
    guess = int(input("Guess the lottery number: "))
    if guess == lottery_number:
        print("You guessed right!")
    else:
        print("You guessed wrong. The lottery number is {}".format(lottery_number))

if __name__ == "__main__":
    main()