#!/bin/bash
pdftk /home/ubuntu/algoTrading/algoTrader/images/demo/*.pdf cat output /home/ubuntu/algoTrading/algoTrader/images/demo.pdf
rm /home/ubuntu/algoTrading/algoTrader/images/demo/*.pdf
echo "Todays drawings" | mail -s 'Demo drawings' lutz.kuenneke89@gmail.com -A /home/ubuntu/algoTrading/algoTrader/images/demo.pdf
pdftk /home/ubuntu/algoTrading/algoTrader/images/live/*.pdf cat output /home/ubuntu/algoTrading/algoTrader/images/live.pdf
rm /home/ubuntu/algoTrading/algoTrader/images/live/*.pdf
echo "Todays drawings" | mail -s 'Live drawings' lutz.kuenneke89@gmail.com -A /home/ubuntu/algoTrading/algoTrader/images/live.pdf
