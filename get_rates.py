from getRates import getRates

cont = getRates.RatesController('../settings.conf')

cont.check_trades()

carrys = cont.check_cfds()

for c in carrys:
    print(c)