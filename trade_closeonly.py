from observer import controller_cython as controller
cont = controller.Controller('/home/tubuntu/settings_triangle.conf', 'live')
allowed_ins = [ins.name for ins in cont.allowed_ins]
margin_ratio = cont.get_margin_ratio()
#if margin_ratio < 0.33: # this corresponds to 25% Margin used ratio
returns = [cont.open_limit(ins, close_only=True) for ins in allowed_ins]
cont_demo = controller.Controller('/home/tubuntu/settings_triangle.conf', 'demo')
margin_ratio = cont_demo.get_margin_ratio()
#if margin_ratio < 0.33: # this corresponds to 25% Margin used ratio
returns = [cont_demo.open_limit(ins, close_only=True) for ins in allowed_ins]
# returns = [cont_demo.simplified_trader(ins) for ins in allowed_ins]
# returns = [cont_demo.open_limit(ins, duration=12, split_position=False, adjust_rr=True) for ins in allowed_ins]
# returns = [cont_demo.simplified_trader(ins) for ins in ['USD_JPY', 'USD_CHF', 'EUR_USD', 'GBP_USD']]
