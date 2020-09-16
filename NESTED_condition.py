n=int(input("enter the value of X ="))
if n==0:
    print('Its zero')
elif n>0:
    if n%2==0:
        print('its even')
    else:
        print('its odd')
elif n<0:
    print('its negative number')
else:
    print("integer")
