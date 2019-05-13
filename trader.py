from observer import controller_cython as controller
cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')
allowed_ins = ['EUR_USD',
               'USD_JPY',
               'USD_CHF',
               'GBP_USD',
               'GBP_JPY',
               'AUD_JPY',
               'USD_CAD',
               'NZD_USD',
               'AUD_USD',
               'EUR_JPY'
               # 'EUR_CHF',
               # 'EUR_AUD',
               # 'EUR_CAD',
               # 'EUR_NZD',
               # 'CHF_JPY',
               # 'CAD_JPY',
               # 'NZD_JPY',
               # 'GBP_CHF',
               # 'AUD_CHF',
               # 'CAD_CHF',
               # 'NZD_CHF',
               # 'GBP_AUD',
               # 'GBP_CAD',
               # 'GBP_NZD',
               # 'AUD_CAD',
               # 'AUD_NZD',
               # 'NZD_CAD',
               # 'EUR_GBP'
               ]
margin_ratio = cont.get_margin_ratio()
returns = [cont.simple_limits(ins, duration=24) for ins in allowed_ins]
cont_demo = controller.Controller('/home/tubuntu/settings_triangle.conf', 'demo')
margin_ratio = cont_demo.get_margin_ratio()
# if margin_ratio < 0.33: # this corresponds to 25% Margin used ratio
allowed_ins = [ins.name for ins in cont_demo.allowed_ins]
returns = [cont_demo.simple_limits(ins, duration=24) for ins in allowed_ins]
# returns = [cont_demo.simplified_trader(ins) for ins in allowed_ins]
# returns = [cont_demo.open_limit(ins, duration=12, split_position=False, adjust_rr=True) for ins in allowed_ins]
# returns = [cont_demo.simplified_trader(ins) for ins in ['USD_JPY', 'USD_CHF', 'EUR_USD', 'GBP_USD']]
