import boto3
import code
import datetime
#ec2 = boto3.resource('ec2')
#instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
#for instance in instances:
#    print(instance.report_status())
def get_current_credits(instance_id):
    client = boto3.client('cloudwatch')
    metrics = client.list_metrics()
    
    cpu_metric = [metric for metric in metrics['Metrics'] if metric['MetricName'] == 'CPUCreditBalance']
    
    response = client.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'cpubalance',
                'MetricStat': {
                    'Metric': { **cpu_metric[0]
                    },
                    'Period': 60,
                    'Stat': 'Average',
                    'Unit': 'Count',
                },
                #'Expression': 'string',
                'Label': 'CPUCreditBalance',
                'ReturnData': True
            },
        ],
        StartTime=(datetime.datetime.now() - datetime.timedelta(minutes = 10)),
        EndTime=datetime.datetime.now())
    try:
        return response['MetricDataResults'][0]['Values'][0]
    except Exception as e:
        print(str(e))
        return None
