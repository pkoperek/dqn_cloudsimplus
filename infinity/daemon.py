import logging
import os
import time
import psycopg2
import signal
from learn import ModelTrainer

log = logging.getLogger('infinity.daemon')
log.setLevel(logging.DEBUG)

root_logger = logging.getLogger('infinity')
console = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s | %(relativeCreated)20d | %(message)s')
console.setFormatter(formatter)
console.setLevel(logging.DEBUG)

root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(console)

def info(message, prefix='infinity daemon'):
    log.info(f'{prefix}: {message}')


is_running = True


def shutdown_handler(signum, frame):
    global is_running
    info('received SIGTERM')
    is_running = False


signal.signal(signal.SIGTERM, shutdown_handler)


def create_connection():
    hostname = os.getenv('POSTGRES_HOST', 'localhost')
    port = int(os.getenv('POSTGRES_PORT', '5432'))
    username = os.getenv('POSTGRES_USERNAME', 'samm')
    password = os.getenv('POSTGRES_PASSWORD', 'samm_secret')
    dbname = os.getenv('POSTGRES_DATABASE', 'samm_db')

    return psycopg2.connect(
        host=hostname,
        port=port,
        user=username,
        password=password,
        dbname=dbname
    )


def store_model(
    cursor,
    model,
    added
):
    cursor.execute(
        '''
            INSERT INTO models (model_data, added)
            VALUES (%s, %s)
            RETURNING model_id
        ''',
        (psycopg2.Binary(model), added)
    )
    new_model_id = cursor.fetchone()[0]
    return new_model_id


def store_model_evaluation(
    connection,
    model,
    evaluation_start_ts,
    evaluation_end_ts,
    evaluation_finish_ts,
    score,
):

    cursor = connection.cursor()
    model_id = store_model(cursor, model, evaluation_finish_ts)

    cursor.execute(
        '''
            INSERT INTO model_evaluations (
                model_id,
                evaluation_start,
                evaluation_end,
                evaluation_finish,
                evaluation_score,
                evaluation_type
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING evaluation_id
        ''',
        (
            model_id,
            evaluation_start_ts,
            evaluation_end_ts,
            evaluation_finish_ts,
            score,
            'SIMULATION'
        )
    )
    evaluation_id = cursor.fetchone()[0]
    connection.commit()
    cursor.close()

    return evaluation_id


def current_timestamp():
    '''
        Current timestamp in count of milliseconds
    '''
    return int(time.time() * 1000)


def count():
    i = 0
    while True:
        yield i
        i += 1


if __name__ == '__main__':
    info('starting')

    connection = create_connection()
    sleep_time = int(os.getenv('SLEEP_TIME', '60'))
    # in s - turned into ms
    simulation_time = int(os.getenv('SIMULATION_TIME', '1800')) * 1000
    # in seconds - turned into ms
    simulation_offset = int(os.getenv('SIMULATION_OFFSET', '120')) * 1000
    reuse_model = os.getenv('MODEL_REUSE', 'false').lower() == 'true'

    model_trainer = None

    for i in count():
        info(f'iteration {i}')

        if not is_running:
            info('shutting down')
            break

        now = current_timestamp()
        simulation_end_timestamp = now - simulation_offset
        simulation_start_timestamp = simulation_end_timestamp - simulation_time

        info(f'training model, reuse: {reuse_model}')
        if not reuse_model or not model_trainer:
            info('creating new model trainer')
            model_trainer = ModelTrainer()

        model, score = model_trainer.train(
            simulation_start_timestamp,
            simulation_end_timestamp
        )
        evaluation_finish_ts = time.time()

        store_model_evaluation(
            connection,
            model.read(),
            simulation_start_timestamp,
            simulation_end_timestamp,
            evaluation_finish_ts,
            score
        )

        info(f'waiting for {sleep_time} sec')
        time.sleep(sleep_time)

    connection.close()
    info('stopped')
