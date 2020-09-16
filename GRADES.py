def computegrade(score):
    if 0.0 <= score <= 1.0:
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        elif score < 0.6:
            return 'F'
        else:
            return 'Bad score'


VALUE = input('Enter score: ')

try:
    score = float(VALUE) #only take float
except:
    print('Bad score') #when enter string etc this part of program wull excute
    quit()

grade = computegrade(score)
print(grade)
