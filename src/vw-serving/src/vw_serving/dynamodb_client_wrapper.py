from boto3.dynamodb.conditions import Key

class DynamoDbClientWrapper:
    def __init__(self, table_session):
        self.ddb_table_session =  table_session

    @staticmethod
    def _dict_to_item(input_json):
        if input_json == '' or not input_json:
            return {
                "NULL": True
            }
        if type(input_json) is dict:
            resp = {}
            for k, v in input_json.items():
                resp[k] = DynamoDbClientWrapper._dict_to_item(v)
            return {
                "M": resp
            }
        if type(input_json) is list:
            list_items = []
            for i in input_json:
                list_items.append(DynamoDbClientWrapper._dict_to_item(i))
            return {
                "L": list_items
            }
        if type(input_json) is str:
            return {
                "S": input_json
            }
        if type(input_json) is int or type(input_json) is float:
            return {
                "N": str(input_json)
            }

    @staticmethod
    def _item_to_dict_helper(input_item):
        if len(input_item) > 1:
            raise ValueError("Length of the value in item is greater than 1. %s" % input_item)
        data_type = next(iter(input_item))
        if data_type == "NULL":
            return ''
        if data_type == "S":
            return input_item[data_type]
        if data_type == "N":
            try:
                return int(input_item[data_type])
            except ValueError:
                return float(input_item[data_type])
        if data_type == "M":
            resp = dict()
            for k, v in input_item[data_type].items():
                resp[k] = DynamoDbClientWrapper._item_to_dict_helper(v)
            return resp
        if data_type == "L":
            resp = []
            for v in input_item[data_type]:
                resp.append(DynamoDbClientWrapper._item_to_dict_helper(v))
            return resp

    @staticmethod
    def _item_to_dict(input_item):
        resp = {}
        for k, v in input_item.items():
            resp[k] = DynamoDbClientWrapper._item_to_dict_helper(v)
        return resp

    """
        Saves a new Experiment to ModelDb, while ensuring that it is not over-writing
        any existing record. This is only called when a new experiment is started.
    """
    def init_new_experiment(self, wf_json):
        self.ddb_table_session.put_item(
            Item=wf_json,
            ConditionExpression='attribute_not_exists(experiment_id) AND attribute_not_exists(model_id)'
        )

    def save_record(self, json_to_save):
        self.ddb_table_session.put_item(
            Item=json_to_save
        )

    def get_workflow_record(self, workflow_id):
        response = self.ddb_table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(workflow_id) & Key('model_id').eq('BASE')
        )
        for i in response['Items']:
            return i
        return None

    def get_model_record(self, experiment_id, model_id):
        response = self.ddb_table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(experiment_id) & Key('model_id').eq(model_id)
        )
        for i in response['Items']:
            return i
        return None

    def deploy_success_status_change(self, workflow_id):
        response = self.ddb_table_session.update_item(
            Key={
                'experiment_id': workflow_id,
                'model_id': "BASE"
            },
            UpdateExpression="set hosting_state=:dpl",
            ExpressionAttributeValues={
                ':dpl': "DEPLOYED"
            }
        )

    def get_next_model_id_and_s3_location(self, workflow_id):
        worfklow_record = self.get_workflow_record(workflow_id)
        if worfklow_record.get('next_hostable_model_id', None) is not None:
            new_model_id = worfklow_record['next_hostable_model_id']
            new_model_record = self.get_model_record(workflow_id, new_model_id)
            return new_model_record['output_s3_location']