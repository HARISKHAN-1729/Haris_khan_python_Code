hrs = input("Enter Hours:")
h = float(hrs)
rate = input("Enter rate per hours")
R = float(rate)
if h<40:
    Pay = h*R
    print(Pay)
if h>40:
    pay = ((40*R)+(h-40)*1.5*R)
    print(pay)
    
