import boto3
from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue
import threading
import time
import uuid
import atexit
import redis
import os
from botocore.exceptions import ClientError


import vw_serving.sagemaker.config.environment as environment
from vw_serving.serve import REDIS_PUBLISHER_CHANNEL


logger = logging.getLogger(__name__)


def encode_data(data, encoding='utf_8'):
    if isinstance(data, bytes):
        return (data.decode(encoding) + '\n').encode(encoding)
    else:
        return str(data + '\n').encode(encoding)


class FirehoseProducer:
    """Basic Firehose Producer.

    Parameters
    ----------
    stream_name : string
        Name of the stream to send the records.
    batch_size : int
        Numbers of records to batch before flushing the queue.
    batch_time : int
        Maximum of seconds to wait before flushing the queue.
    max_retries: int
        Maximum number of times to retry the put operation.
    firehose_client: boto3.client
        Firehose client.

    Attributes
    ----------
    records : array
        Queue of formated records.
    pool: concurrent.futures.ThreadPoolExecutor
        Pool of threads handling client I/O.
    """

    def __init__(self, stream_name, batch_size=50,
                 batch_time=.2, max_retries=5, threads=2,
                 firehose_client=None):
        self.stream_name = stream_name
        self.buffer_on = os.getenv(environment.FIREHOSE_BUFFER_ON, 'false').lower() == 'true'

        self.max_retries = max_retries
        if firehose_client is None:
            firehose_client = boto3.client('firehose')
        self.firehose_client = firehose_client
        self.pool = ThreadPoolExecutor(threads)

        if self.buffer_on:
            self.queue = Queue()
            self.batch_size = batch_size
            self.batch_time = batch_time
            self.last_flush = time.time()
            self.monitor_running = threading.Event()
            self.monitor_running.set()
            self.pool.submit(self.monitor)
            logger.info(f"Buffering data with batch_size {self.batch_size} and batch_time {self.batch_time}s before push to Firehose")
        else:
            logger.info("Write data directly to Firehose without batching")

        atexit.register(self.close)

    def monitor(self):
        """Flushes the queue periodically."""
        while self.monitor_running.is_set():
            if time.time() - self.last_flush > self.batch_time:
                if not self.queue.empty():
                    logger.info(f"Queue Flush: Data flushed to Firehose stream: {self.stream_name}")
                    self.flush_queue()
            time.sleep(self.batch_time)

    def put_records(self, records):
        """Add a list of data records to the record queue in the proper format.
        Convinience method that calls self.put_record for each element.

        Parameters
        ----------
        records : list
            Lists of records to send.

        """
        for record in records:
            self.put_record(record)

    def put_record(self, data, pool_submit=True):
        """Add data to the record queue in the proper format.

        Parameters
        ----------
        data : str
            Data to send.

        """
        # Byte encode the data
        data = encode_data(data)

        # Build the record
        record = {
            'Data': data
        }
        if self.buffer_on:
            # Flush the queue if it reaches the batch size
            if self.queue.qsize() >= self.batch_size:
                logger.info("Queue Flush: batch size reached")
                self.pool.submit(self.flush_queue)

            # Append the record
            logger.debug('Putting record "{}"'.format(record['Data'][:100]))
            self.queue.put(record)
        else:
            if pool_submit:
                self.pool.submit(self.send_record, record)
            else:
                self.send_record(record)

    def close(self):
        """Flushes the queue and waits for the executor to finish."""
        logger.info('Closing producer')
        self.flush_queue()
        self.monitor_running.clear()
        self.pool.shutdown()
        logger.info('Producer closed')

    def flush_queue(self):
        """Grab all the current records in the queue and send them."""
        records = []

        while not self.queue.empty() and len(records) < self.batch_size:
            records.append(self.queue.get())

        if records:
            self.send_records(records)
            self.last_flush = time.time()
        
    def listen_to_redis_channel(self, channel):
        redis_client = redis.Redis()
        pubsub = redis_client.pubsub()
        pubsub.subscribe(channel)
        logger.info(f"Listening to redis channel: {channel}")
        for item in pubsub.listen():
            if item['type'] == 'message':
                self.put_record(item['data'])

    def send_records(self, records, attempt=0):
        """Send records to the Firehose stream.

        Falied records are sent again with an exponential backoff decay.

        Parameters
        ----------
        records : array
            Array of formated records to send.
        attempt: int
            Number of times the records have been sent without success.
        """

        # If we already tried more times than we wanted, save to a file
        if attempt > self.max_retries:
            logger.warning('Writing {} records to file'.format(len(records)))
            return

        # Sleep before retrying
        if attempt:
            time.sleep(2 ** attempt * .1)

        response = self.firehose_client.put_record_batch(DeliveryStreamName=self.stream_name,
                                                         Records=records)
        failed_record_count = response['FailedPutCount']

        # Grab failed records
        if failed_record_count:
            logger.warning('Retrying failed records')
            failed_records = []
            for i, record in enumerate(response['RequestResponses']):
                if record.get('ErrorCode'):
                    failed_records.append(records[i])

            # Recursive call
            attempt += 1
            self.send_records(failed_records, attempt=attempt)
    
    def send_record(self, record, attempt=0):
        """Send single record to the Firehose stream.

        Falied records are sent again with an exponential backoff decay.

        Parameters
        ----------
        record : array
            Formated record to send.
        attempt: int
            Number of times the record have been sent without success.
        """

        # If we already tried more times than we wanted, save to a file
        if attempt > self.max_retries:
            logger.warning('Writing {} records to file'.format(len(record)))
            return

        # Sleep before retrying
        if attempt:
            time.sleep(2 ** attempt * .1)

        try:
            self.firehose_client.put_record(DeliveryStreamName=self.stream_name,
                                            Record=record)
        except ClientError as e:
            logger.warning('Retrying failed records')
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']
            logger.warning(f'{error_code}:{message}')
            attempt += 1
            self.send_record(record, attempt=attempt)


def main():
    producer = FirehoseProducer("obs_rewards_delivery_stream")
    producer.listen_to_redis_channel(channel=REDIS_PUBLISHER_CHANNEL)


if __name__ == "__main__":
    main()