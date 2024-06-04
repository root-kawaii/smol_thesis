// 10 Implant
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "nordic_common.h"
#include "nrf.h"
#include "ble_hci.h"
#include "ble_advdata.h"
#include "ble_advertising.h"
#include "ble_conn_params.h"
#include "nrf_sdh.h"
#include "nrf_sdh_soc.h"
#include "nrf_sdh_ble.h"
#include "nrf_ble_gatt.h"
#include "nrf_ble_qwr.h"
#include "app_timer.h"
#include "ble_nus.h"
#include "app_uart.h"
#include "app_util_platform.h"
#include "bsp_btn_ble.h"
#include "nrf_pwr_mgmt.h"
#include "nrfx_timer.h"

/* in più rispetto all'esempio UART */
#include "nrf_delay.h"
#include "ringbuf.h"
#include "nrf_drv_spi.h"
#include "nrf_drv_gpiote.h"
// aggiunti per implementare il peer manager
#include "peer_manager.h"
#include "peer_manager_handler.h"
#include "fds.h"
#include "nrf_ble_lesc.h"

#if defined(UART_PRESENT)
#include "nrf_uart.h"
#endif
#if defined(UARTE_PRESENT)
#include "nrf_uarte.h"
#endif

#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"

#define APP_BLE_CONN_CFG_TAG 1 /**< A tag identifying the SoftDevice BLE configuration. */

#define DEVICE_NAME "Senseback"                          /**< Name of device. Will be included in the advertising data. */
#define NUS_SERVICE_UUID_TYPE BLE_UUID_TYPE_VENDOR_BEGIN /**< UUID type for the Nordic UART Service (vendor specific). */

#define APP_BLE_OBSERVER_PRIO 3 /**< Application's BLE observer priority. You shouldn't need to modify this value. */

#define APP_ADV_INTERVAL 64 /**< The advertising interval (in units of 0.625 ms. This value corresponds to 40 ms). */

#define APP_ADV_DURATION 18000 /**< The advertising duration (180 seconds) in units of 10 milliseconds. */

#define MIN_CONN_INTERVAL MSEC_TO_UNITS(50, UNIT_1_25_MS)    /**< Minimum acceptable connection interval (50 ms). */
#define MAX_CONN_INTERVAL MSEC_TO_UNITS(50, UNIT_1_25_MS)    /**< Maximum acceptable connection interval (50 ms). */
#define SLAVE_LATENCY 0                                      /**< Slave latency. */
#define CONN_SUP_TIMEOUT MSEC_TO_UNITS(8010, UNIT_10_MS)     /**< Connection supervisory timeout (4 seconds), Supervision Timeout uses 10 ms units. */
#define FIRST_CONN_PARAMS_UPDATE_DELAY APP_TIMER_TICKS(5000) /**< Time from initiating event (connect or start of notification) to first time sd_ble_gap_conn_param_update is called (5 seconds). */
#define NEXT_CONN_PARAMS_UPDATE_DELAY APP_TIMER_TICKS(30000) /**< Time between each call to sd_ble_gap_conn_param_update after the first call (30 seconds). */
#define MAX_CONN_PARAMS_UPDATE_COUNT 3                       /**< Number of attempts before giving up the connection parameter negotiation. */

#define LESC_DEBUG_MODE 1 /**< Set to 1 to use LESC debug keys, allows you to use a sniffer to inspect traffic. */

#define SEC_PARAM_BOND 1                               /**< Perform bonding. */
#define SEC_PARAM_MITM 0                               /**< Man In The Middle protection not required. */
#define SEC_PARAM_LESC 1                               /**< LE Secure Connections enabled. */
#define SEC_PARAM_KEYPRESS 0                           /**< Keypress notifications not enabled. */
#define SEC_PARAM_IO_CAPABILITIES BLE_GAP_IO_CAPS_NONE /**< No I/O capabilities. */
#define SEC_PARAM_OOB 0                                /**< Out Of Band data not available. */
#define SEC_PARAM_MIN_KEY_SIZE 7                       /**< Minimum encryption key size. */
#define SEC_PARAM_MAX_KEY_SIZE 16                      /**< Maximum encryption key size. */

#define DEAD_BEEF 0xDEADBEEF /**< Value used as error code on stack dump, can be used to identify stack location on stack unwind. */

BLE_NUS_DEF(m_nus, NRF_SDH_BLE_TOTAL_LINK_COUNT); /**< BLE NUS service instance. */
NRF_BLE_GATT_DEF(m_gatt);                         /**< GATT module instance. */
NRF_BLE_QWR_DEF(m_qwr);                           /**< Context for the Queued Write module.*/
BLE_ADVERTISING_DEF(m_advertising);               /**< Advertising module instance. */
APP_TIMER_DEF(m_test_timer_id);

static uint16_t m_conn_handle = BLE_CONN_HANDLE_INVALID;               /**< Handle of the current connection. */
static uint16_t m_ble_nus_max_data_len = BLE_GATT_ATT_MTU_DEFAULT - 3; /**< Maximum length of data (in bytes) that can be transmitted to the peer by the Nordic UART service module. */
static ble_uuid_t m_adv_uuids[] =                                      /**< Universally unique service identifier. */
    {
        {BLE_UUID_NUS_SERVICE, NUS_SERVICE_UUID_TYPE}};

// TIMER
const nrfx_timer_t timer1 = NRFX_TIMER_INSTANCE(1);
const nrfx_timer_t timer2 = NRFX_TIMER_INSTANCE(2);
#define TIMER1_INTERVAL_US 18.2
#define TIMER2_INTERVAL_MS 1

// BUFFER
#define UART_RING_SIZE 16384

struct ringbuf uartRx;
static uint16_t ringBuffer[UART_RING_SIZE];

#define SENSEBACK_MTU 244

static uint8_t nusTx[SENSEBACK_MTU];

#define DATA_TO_SEND 32400

static volatile bool txActive = false;
static volatile bool bleTxBusy = false;
static volatile uint16_t txRdPtr;
static volatile uint16_t txSize;

static volatile uint16_t data_sent = 0;
static volatile uint16_t data_arrived = 0;
static volatile uint16_t CNT = 0;
static volatile bool transmission = false;
static volatile uint16_t wait = 0;

static volatile uint16_t time_elapsed_ms = 0;
static volatile bool first = true;

#define TX_POWER_LEVEL 0

void notification_send()
{
    uint16_t length;
    uint32_t err_code = NRF_SUCCESS;
    while (err_code == NRF_SUCCESS && txRdPtr < txSize)
    {
        length = ((txSize - txRdPtr) > SENSEBACK_MTU) ? SENSEBACK_MTU : (txSize - txRdPtr);

        err_code = ble_nus_data_send(&m_nus, &nusTx[txRdPtr], &length, m_conn_handle);

        if ((err_code != NRF_ERROR_INVALID_STATE) &&
            (err_code != NRF_ERROR_RESOURCES) &&
            (err_code != NRF_ERROR_NOT_FOUND))
        {
            APP_ERROR_CHECK(err_code);
        }
        if (err_code == NRF_SUCCESS)
        {
            txRdPtr += length;
            data_sent += length;

            if ((nusTx[txRdPtr - 1] == 255) && ((nusTx[txRdPtr - 2] == 255)))
            {
                uint32_t ticksVal = nrfx_timer_capture(&timer2, NRF_TIMER_CC_CHANNEL2);
                float tVal = (float)ticksVal * 0.0005;
                float time_elapsed = (float)time_elapsed_ms + tVal;
                NRF_LOG_INFO("Time elapsed total: " NRF_LOG_FLOAT_MARKER " ms", NRF_LOG_FLOAT(time_elapsed));
                NRF_LOG_INFO("Data sent: %u", data_sent);
                float datarate = (float)data_sent * 8 / (time_elapsed);
                NRF_LOG_INFO("Datarate: " NRF_LOG_FLOAT_MARKER " bits per ms", NRF_LOG_FLOAT(datarate));
                data_sent = 0;
                NRF_LOG_INFO("Waits: %u", wait);
            }
        }
        if (err_code == NRF_ERROR_RESOURCES)
        {
            bleTxBusy = true;
            wait++;
        }
    }
    if (txRdPtr >= txSize)
        txActive = false;
}

void on_tx_complete()
{
    if (bleTxBusy)
    {
        bleTxBusy = false;
        notification_send();
    }
}

/**@brief Function for assert macro callback.
 *
 * @details This function will be called in case of an assert in the SoftDevice.
 *
 * @warning This handler is an example only and does not fit a final product. You need to analyse
 *          how your product is supposed to react in case of Assert.
 * @warning On assert from the SoftDevice, the system can only recover on reset.
 *
 * @param[in] line_num    Line number of the failing ASSERT call.
 * @param[in] p_file_name File name of the failing ASSERT call.
 */
void assert_nrf_callback(uint16_t line_num, const uint8_t *p_file_name)
{
    app_error_handler(DEAD_BEEF, line_num, p_file_name);
}

/**@brief Clear bond information from persistent storage.
 */
static void delete_bonds(void)
{
    ret_code_t err_code;

    NRF_LOG_INFO("Erase bonds!");

    err_code = pm_peers_delete();
    APP_ERROR_CHECK(err_code);
}

/**@brief Function for starting advertising.
 */
void advertising_start(bool erase_bonds)
{
    if (erase_bonds == true)
    {
        delete_bonds();
        // Advertising is started by PM_EVT_PEERS_DELETE_SUCCEEDED event.
    }
    else
    {
        ret_code_t err_code;

        err_code = ble_advertising_start(&m_advertising, BLE_ADV_MODE_FAST);
        APP_ERROR_CHECK(err_code);
    }
}

/**@brief Function for handling Peer Manager events.
 *
 * @param[in] p_evt  Peer Manager event.
 */
static void pm_evt_handler(pm_evt_t const *p_evt)
{
    pm_handler_on_pm_evt(p_evt);
    pm_handler_flash_clean(p_evt);

    switch (p_evt->evt_id)
    {
    case PM_EVT_PEERS_DELETE_SUCCEEDED:
        advertising_start(false);
        break;

    case PM_EVT_BONDED_PEER_CONNECTED:
        NRF_LOG_INFO("Bonded peer connected successfully!");
        break;

    default:
        break;
    }
}

/**@brief Function for handling the interrupts due to the timer1 overflow --> 12.5us
    Simulating the arrival of the data at 80kHz
 */
static void timer1_handler(nrf_timer_event_t event_type, void *p_context)
{
    switch (event_type)
    {
    case NRF_TIMER_EVENT_COMPARE1:
        if (transmission)
        {
            CNT++;
            ringbuf_put(&uartRx, CNT);
            data_arrived++;
            if (first)
            {
                first = false;
                nrfx_timer_enable(&timer2);
                time_elapsed_ms = 0;
            }
            if (data_arrived > (((DATA_TO_SEND - 2) / 2) - 1))
            {
                data_arrived = 0;
                CNT = 0;
                ringbuf_put(&uartRx, 65535);
                transmission = false;
                first = true;
            }
        }
        break;

    default:
        break;
    }
}

/**@brief Function for handling the interrupts due to the timer2 overflow --> 1us
    To calculate the throughput
 */
static void timer2_handler(nrf_timer_event_t event_type, void *p_context)
{
    switch (event_type)
    {
    case NRF_TIMER_EVENT_COMPARE2:
        time_elapsed_ms += 1;
        break;

    default:
        break;
    }
}

/**@brief Function for initializing the timer module.
 */
static void timers_init(void)
{
    ret_code_t err_code = app_timer_init();
    APP_ERROR_CHECK(err_code);

    // First timer --> to handle the sending of data
    nrfx_timer_config_t timer1_config = NRFX_TIMER_DEFAULT_CONFIG;
    timer1_config.frequency = NRF_TIMER_FREQ_2MHz;
    err_code = nrfx_timer_init(&timer1, &timer1_config, timer1_handler);
    APP_ERROR_CHECK(err_code);

    uint32_t time_ticks = nrfx_timer_us_to_ticks(&timer1, TIMER1_INTERVAL_US) + 1;
    NRF_LOG_DEBUG("ticks timer1: %u", time_ticks);

    nrfx_timer_extended_compare(&timer1, NRF_TIMER_CC_CHANNEL1, time_ticks, NRF_TIMER_SHORT_COMPARE1_CLEAR_MASK, true);

    // Second timer --> to calculate datarate
    nrfx_timer_config_t timer2_config = NRFX_TIMER_DEFAULT_CONFIG;
    timer2_config.frequency = NRF_TIMER_FREQ_2MHz;
    err_code = nrfx_timer_init(&timer2, &timer2_config, timer2_handler);
    APP_ERROR_CHECK(err_code);

    time_ticks = nrfx_timer_ms_to_ticks(&timer2, TIMER2_INTERVAL_MS);
    NRF_LOG_DEBUG("ticks timer2: %u", time_ticks);

    nrfx_timer_extended_compare(&timer2, NRF_TIMER_CC_CHANNEL2, time_ticks, NRF_TIMER_SHORT_COMPARE2_CLEAR_MASK, true);
}

/**@brief Function for the GAP initialization.
 *
 * @details This function will set up all the necessary GAP (Generic Access Profile) parameters of
 *          the device. It also sets the permissions and appearance.
 */
static void gap_params_init(void)
{
    uint32_t err_code;
    ble_gap_conn_params_t gap_conn_params;
    ble_gap_conn_sec_mode_t sec_mode;

    BLE_GAP_CONN_SEC_MODE_SET_ENC_NO_MITM(&sec_mode);

    err_code = sd_ble_gap_device_name_set(&sec_mode,
                                          (const uint8_t *)DEVICE_NAME,
                                          strlen(DEVICE_NAME));
    APP_ERROR_CHECK(err_code);

    memset(&gap_conn_params, 0, sizeof(gap_conn_params));

    gap_conn_params.min_conn_interval = MIN_CONN_INTERVAL;
    gap_conn_params.max_conn_interval = MAX_CONN_INTERVAL;
    gap_conn_params.slave_latency = SLAVE_LATENCY;
    gap_conn_params.conn_sup_timeout = CONN_SUP_TIMEOUT;

    err_code = sd_ble_gap_ppcp_set(&gap_conn_params);
    APP_ERROR_CHECK(err_code);
}

/**@brief Function for handling Queued Write Module errors.
 *
 * @details A pointer to this function will be passed to each service which may need to inform the
 *          application about an error.
 *
 * @param[in]   nrf_error   Error code containing information about what went wrong.
 */
static void nrf_qwr_error_handler(uint32_t nrf_error)
{
    APP_ERROR_HANDLER(nrf_error);
}

// function to initalize peer manager and set peer manager security parameters
static void peer_manager_init(void)
{
    ble_gap_sec_params_t sec_param;
    ret_code_t err_code;

    err_code = pm_init();
    APP_ERROR_CHECK(err_code);

    memset(&sec_param, 0, sizeof(ble_gap_sec_params_t));

    // Security parameters to be used for all security procedures.
    sec_param.bond = SEC_PARAM_BOND;
    sec_param.mitm = SEC_PARAM_MITM;
    sec_param.lesc = SEC_PARAM_LESC;
    sec_param.keypress = SEC_PARAM_KEYPRESS;
    sec_param.io_caps = SEC_PARAM_IO_CAPABILITIES;
    sec_param.oob = SEC_PARAM_OOB;
    sec_param.min_key_size = SEC_PARAM_MIN_KEY_SIZE;
    sec_param.max_key_size = SEC_PARAM_MAX_KEY_SIZE;
    sec_param.kdist_own.enc = 1;
    sec_param.kdist_own.id = 1;
    sec_param.kdist_peer.enc = 1;
    sec_param.kdist_peer.id = 1;

    err_code = pm_sec_params_set(&sec_param);
    APP_ERROR_CHECK(err_code);

    err_code = pm_register(pm_evt_handler);
    APP_ERROR_CHECK(err_code);
}

/**@brief Function for handling the data from the Nordic UART Service.
 *
 * @details This function will process the data received from the Nordic UART BLE Service and send
 *          it to the UART module.
 *
 * @param[in] p_evt       Nordic UART Service event.
 */
/**@snippet [Handling the data received over BLE] */
static void nus_data_handler(ble_nus_evt_t *p_evt)
{
    if (p_evt->type == BLE_NUS_EVT_RX_DATA)
    {
        uint32_t err_code;

        /*aggiunto da me per salvare localmente il comando ricevuto in p_data*/
        char dongle_command[BLE_NUS_MAX_DATA_LEN];
        memset(dongle_command, 0, BLE_NUS_MAX_DATA_LEN);
        memcpy(dongle_command, p_evt->params.rx_data.p_data, p_evt->params.rx_data.length);
        /*aggiunto da me per salvare localmente il comando ricevuto in p_data*/

        NRF_LOG_DEBUG("Received data from BLE NUS. Writing data on UART.");
        NRF_LOG_HEXDUMP_DEBUG(p_evt->params.rx_data.p_data, p_evt->params.rx_data.length);

        if (strcmp(dongle_command, "START!") == 0)
        {
            transmission = true;
            nrfx_timer_enable(&timer1);
            time_elapsed_ms = 0;
            memcpy(dongle_command, 0, 1);
        }
        else if (strcmp(dongle_command, "RESET!") == 0)
        {
            CNT = 0;
        }
        else if (strcmp(dongle_command, "STOP!") == 0)
        {
            nrfx_timer_disable(&timer1);
        }
    }
}
/**@snippet [Handling the data received over BLE] */

/**@brief Function for initializing services that will be used by the application.
 */
static void services_init(void)
{
    NRF_LOG_INFO("services init enter");
    uint32_t err_code;
    ble_nus_init_t nus_init;
    nrf_ble_qwr_init_t qwr_init = {0};

    // Initialize Queued Write Module.
    qwr_init.error_handler = nrf_qwr_error_handler;

    err_code = nrf_ble_qwr_init(&m_qwr, &qwr_init);
    APP_ERROR_CHECK(err_code);

    // Initialize peer manager
    peer_manager_init();

    // Initialize NUS.
    memset(&nus_init, 0, sizeof(nus_init));

    nus_init.data_handler = nus_data_handler;

    err_code = ble_nus_init(&m_nus, &nus_init);
    APP_ERROR_CHECK(err_code);
}

/**@brief Function for handling an event from the Connection Parameters Module.
 *
 * @details This function will be called for all events in the Connection Parameters Module
 *          which are passed to the application.
 *
 * @note All this function does is to disconnect. This could have been done by simply setting
 *       the disconnect_on_fail config parameter, but instead we use the event handler
 *       mechanism to demonstrate its use.
 *
 * @param[in] p_evt  Event received from the Connection Parameters Module.
 */
static void on_conn_params_evt(ble_conn_params_evt_t *p_evt)
{
    uint32_t err_code;

    if (p_evt->evt_type == BLE_CONN_PARAMS_EVT_FAILED)
    {
        err_code = sd_ble_gap_disconnect(m_conn_handle, BLE_HCI_CONN_INTERVAL_UNACCEPTABLE);
        APP_ERROR_CHECK(err_code);
    }
}

/**@brief Function for handling errors from the Connection Parameters module.
 *
 * @param[in] nrf_error  Error code containing information about what went wrong.
 */
static void conn_params_error_handler(uint32_t nrf_error)
{
    APP_ERROR_HANDLER(nrf_error);
}

/**@brief Function for initializing the Connection Parameters module.
 */
static void conn_params_init(void)
{
    uint32_t err_code;
    ble_conn_params_init_t cp_init;

    memset(&cp_init, 0, sizeof(cp_init));

    cp_init.p_conn_params = NULL;
    cp_init.first_conn_params_update_delay = FIRST_CONN_PARAMS_UPDATE_DELAY;
    cp_init.next_conn_params_update_delay = NEXT_CONN_PARAMS_UPDATE_DELAY;
    cp_init.max_conn_params_update_count = MAX_CONN_PARAMS_UPDATE_COUNT;
    cp_init.start_on_notify_cccd_handle = BLE_GATT_HANDLE_INVALID;
    cp_init.disconnect_on_fail = false;
    cp_init.evt_handler = on_conn_params_evt;
    cp_init.error_handler = conn_params_error_handler;

    err_code = ble_conn_params_init(&cp_init);
    APP_ERROR_CHECK(err_code);
}

/* *************************************************************************** */

/**@brief Function for putting the chip into sleep mode.
 *
 * @note This function will not return.
 */
static void sleep_mode_enter(void)
{
    uint32_t err_code = bsp_indication_set(BSP_INDICATE_IDLE);
    APP_ERROR_CHECK(err_code);

    // Prepare wakeup buttons.
    err_code = bsp_btn_ble_sleep_mode_prepare();
    APP_ERROR_CHECK(err_code);

    // Go to system-off mode (this function will not return; wakeup will cause a reset).
    err_code = sd_power_system_off();
    APP_ERROR_CHECK(err_code);
}

/* *************************************************************************** */

/**@brief Function for handling advertising events.
 *
 * @details This function will be called for advertising events which are passed to the application.
 *
 * @param[in] ble_adv_evt  Advertising event.
 */
static void on_adv_evt(ble_adv_evt_t ble_adv_evt)
{
    uint32_t err_code;

    switch (ble_adv_evt)
    {
    case BLE_ADV_EVT_FAST:
        err_code = bsp_indication_set(BSP_INDICATE_ADVERTISING);
        APP_ERROR_CHECK(err_code);
        break;
    case BLE_ADV_EVT_IDLE:
        sleep_mode_enter();
        break;
    default:
        break;
    }
}

/**@brief Function for handling BLE events.
 *
 * @param[in]   p_ble_evt   Bluetooth stack event.
 * @param[in]   p_context   Unused.
 */
static void ble_evt_handler(ble_evt_t const *p_ble_evt, void *p_context)
{
    uint32_t err_code;
    ble_gap_evt_t const *p_gap_evt = &p_ble_evt->evt.gap_evt;

    switch (p_ble_evt->header.evt_id)
    {
    case BLE_GAP_EVT_CONNECTED:
        NRF_LOG_INFO("Connected");
        err_code = bsp_indication_set(BSP_INDICATE_CONNECTED);
        APP_ERROR_CHECK(err_code);
        m_conn_handle = p_ble_evt->evt.gap_evt.conn_handle;
        err_code = nrf_ble_qwr_conn_handle_assign(&m_qwr, m_conn_handle);
        APP_ERROR_CHECK(err_code);
        ble_gap_phys_t const phys =
            {
                .rx_phys = BLE_GAP_PHY_2MBPS,
                .tx_phys = BLE_GAP_PHY_2MBPS,
            };
        err_code = sd_ble_gap_phy_update(p_gap_evt->conn_handle, &phys);
        APP_ERROR_CHECK(err_code);

        break;

    case BLE_GAP_EVT_DISCONNECTED:
        NRF_LOG_INFO("Disconnected");
        // LED indication will be changed when advertising starts.
        m_conn_handle = BLE_CONN_HANDLE_INVALID;
        break;

    case BLE_GAP_EVT_PHY_UPDATE_REQUEST:
    {
        NRF_LOG_DEBUG("PHY update request.");
        ble_gap_phys_t const phys =
            {
                .rx_phys = BLE_GAP_PHY_AUTO,
                .tx_phys = BLE_GAP_PHY_AUTO,
            };
        err_code = sd_ble_gap_phy_update(p_ble_evt->evt.gap_evt.conn_handle, &phys);
        APP_ERROR_CHECK(err_code);
    }
    break;

    case BLE_GATTC_EVT_TIMEOUT:
        // Disconnect on GATT Client timeout event.
        err_code = sd_ble_gap_disconnect(p_ble_evt->evt.gattc_evt.conn_handle,
                                         BLE_HCI_REMOTE_USER_TERMINATED_CONNECTION);
        APP_ERROR_CHECK(err_code);
        break;

    case BLE_GATTS_EVT_TIMEOUT:
        // Disconnect on GATT Server timeout event.
        err_code = sd_ble_gap_disconnect(p_ble_evt->evt.gatts_evt.conn_handle,
                                         BLE_HCI_REMOTE_USER_TERMINATED_CONNECTION);
        APP_ERROR_CHECK(err_code);
        break;

    case BLE_GATTS_EVT_HVN_TX_COMPLETE:
        on_tx_complete();
        break;
    case BLE_GAP_EVT_SEC_PARAMS_REQUEST:
        NRF_LOG_DEBUG("BLE_GAP_EVT_SEC_PARAMS_REQUEST");
        break;

    case BLE_GAP_EVT_AUTH_KEY_REQUEST:
        NRF_LOG_INFO("BLE_GAP_EVT_AUTH_KEY_REQUEST");
        break;

    case BLE_GAP_EVT_LESC_DHKEY_REQUEST:
        NRF_LOG_INFO("BLE_GAP_EVT_LESC_DHKEY_REQUEST");
        break;
        // show authentication state
    case BLE_GAP_EVT_AUTH_STATUS:
        NRF_LOG_INFO("BLE_GAP_EVT_AUTH_STATUS: status=0x%x bond=0x%x lv4: %d kdist_own:0x%x kdist_peer:0x%x",
                     p_ble_evt->evt.gap_evt.params.auth_status.auth_status,
                     p_ble_evt->evt.gap_evt.params.auth_status.bonded,
                     p_ble_evt->evt.gap_evt.params.auth_status.sm1_levels.lv4,
                     *((uint8_t *)&p_ble_evt->evt.gap_evt.params.auth_status.kdist_own),
                     *((uint8_t *)&p_ble_evt->evt.gap_evt.params.auth_status.kdist_peer));
        if (p_ble_evt->evt.gap_evt.params.auth_status.auth_status == BLE_GAP_SEC_STATUS_SUCCESS)
        {
            transmission = true;
            nrfx_timer_enable(&timer1);
            time_elapsed_ms = 0;
        }
        break;

    default:
        // No implementation needed.
        break;
    }
}

/**@brief Function for the SoftDevice initialization.
 *
 * @details This function initializes the SoftDevice and the BLE event interrupt.
 */
static void ble_stack_init(void)
{
    ret_code_t err_code;

    err_code = nrf_sdh_enable_request();
    APP_ERROR_CHECK(err_code);

    // Configure the BLE stack using the default settings.
    // Fetch the start address of the application RAM.
    uint32_t ram_start = 0;
    err_code = nrf_sdh_ble_default_cfg_set(APP_BLE_CONN_CFG_TAG, &ram_start);
    APP_ERROR_CHECK(err_code);

    /*
    ble_cfg_t ble_cfg;
    memset(&ble_cfg, 0, sizeof ble_cfg);
    ble_cfg.conn_cfg.conn_cfg_tag = APP_BLE_CONN_CFG_TAG;
    ble_cfg.conn_cfg.params.gatts_conn_cfg.hvn_tx_queue_size = 10;
    err_code = sd_ble_cfg_set(BLE_CONN_CFG_GATTS, &ble_cfg, ram_start);
    APP_ERROR_CHECK(err_code);
    */

    // Enable BLE stack.
    err_code = nrf_sdh_ble_enable(&ram_start);
    APP_ERROR_CHECK(err_code);

    // Register a handler for BLE events.
    NRF_SDH_BLE_OBSERVER(m_ble_observer, APP_BLE_OBSERVER_PRIO, ble_evt_handler, NULL);
}

/**@brief Function for handling events from the GATT library. */
void gatt_evt_handler(nrf_ble_gatt_t *p_gatt, nrf_ble_gatt_evt_t const *p_evt)
{
    if ((m_conn_handle == p_evt->conn_handle) && (p_evt->evt_id == NRF_BLE_GATT_EVT_ATT_MTU_UPDATED))
    {
        m_ble_nus_max_data_len = p_evt->params.att_mtu_effective - OPCODE_LENGTH - HANDLE_LENGTH;
        NRF_LOG_INFO("Data len is set to 0x%X(%d)", m_ble_nus_max_data_len, m_ble_nus_max_data_len);
    }
    NRF_LOG_DEBUG("ATT MTU exchange completed. central 0x%x peripheral 0x%x",
                  p_gatt->att_mtu_desired_central,
                  p_gatt->att_mtu_desired_periph);
}

/**@brief Function for initializing the GATT library. */
void gatt_init(void)
{
    ret_code_t err_code;

    err_code = nrf_ble_gatt_init(&m_gatt, gatt_evt_handler);
    APP_ERROR_CHECK(err_code);

    err_code = nrf_ble_gatt_att_mtu_periph_set(&m_gatt, NRF_SDH_BLE_GATT_MAX_MTU_SIZE);
    APP_ERROR_CHECK(err_code);
}

/**@brief Function for initializing the Advertising functionality.
 */
static void advertising_init(void)
{
    uint32_t err_code;
    ble_advertising_init_t init;

    memset(&init, 0, sizeof(init));

    init.advdata.name_type = BLE_ADVDATA_FULL_NAME;
    init.advdata.include_appearance = false;
    init.advdata.flags = BLE_GAP_ADV_FLAGS_LE_ONLY_LIMITED_DISC_MODE;

    init.srdata.uuids_complete.uuid_cnt = sizeof(m_adv_uuids) / sizeof(m_adv_uuids[0]);
    init.srdata.uuids_complete.p_uuids = m_adv_uuids;

    init.config.ble_adv_fast_enabled = true;
    init.config.ble_adv_fast_interval = APP_ADV_INTERVAL;
    init.config.ble_adv_fast_timeout = APP_ADV_DURATION;
    init.evt_handler = on_adv_evt;

    err_code = ble_advertising_init(&m_advertising, &init);
    APP_ERROR_CHECK(err_code);

    ble_advertising_conn_cfg_tag_set(&m_advertising, APP_BLE_CONN_CFG_TAG);
}

/**@brief Function for initializing the nrf log module.
 */
static void log_init(void)
{
    ret_code_t err_code = NRF_LOG_INIT(NULL);
    APP_ERROR_CHECK(err_code);

    NRF_LOG_DEFAULT_BACKENDS_INIT();
}

/**@brief Function for initializing power management.
 */
static void power_management_init(void)
{
    ret_code_t err_code;
    err_code = nrf_pwr_mgmt_init();
    APP_ERROR_CHECK(err_code);
}

static void spiBuffProcess()
{
    uint16_t element;

    // Deal with the data going to the chip
    //  check other codes

    // Deal with the data coming from the chip

    if (!txActive)
    {
        uint16_t queueLength = ringbuf_elements(&uartRx);
        if (queueLength >= (SENSEBACK_MTU / 2))
        {
            for (int i = 0; i < (SENSEBACK_MTU / 2); i++)
            {
                element = ringbuf_get(&uartRx);
                nusTx[i * 2] = (uint8_t)element;
                nusTx[i * 2 + 1] = element >> 8;
            }

            txActive = true;
            txSize = SENSEBACK_MTU;
            txRdPtr = 0;
            notification_send();
        }
        else if ((queueLength > 0) && (data_sent > (DATA_TO_SEND - SENSEBACK_MTU)))
        {
            NRF_LOG_INFO("Last packet size /2: %u", queueLength);
            for (int i = 0; i < queueLength; i++)
            {
                element = ringbuf_get(&uartRx);
                nusTx[i * 2] = (uint8_t)element;
                nusTx[i * 2 + 1] = element >> 8;
            }

            txActive = true;
            txSize = 2 * queueLength;
            txRdPtr = 0;
            notification_send();
        }
    }
}

/**@brief Function for handling the idle state (main loop).
 *
 * @details If there is no pending log operation, then sleep until next the next event occurs.
 */
static void idle_state_handle(void)
{
    spiBuffProcess();
    UNUSED_RETURN_VALUE(NRF_LOG_PROCESS());
    nrf_pwr_mgmt_run();
}

/* Connection ext len ON */
void conn_evt_len_ext_set(void)
{
    ret_code_t err_code;
    ble_opt_t opt;

    memset(&opt, 0x00, sizeof(opt));
    opt.common_opt.conn_evt_ext.enable = 1;

    err_code = sd_ble_opt_set(BLE_COMMON_OPT_CONN_EVT_EXT, &opt);
    APP_ERROR_CHECK(err_code);
}
/* Connection ext len ON */

/* TX power set */
static void tx_power_set(void)
{
    ret_code_t err_code = sd_ble_gap_tx_power_set(BLE_GAP_TX_POWER_ROLE_ADV, m_advertising.adv_handle, TX_POWER_LEVEL);
    APP_ERROR_CHECK(err_code);
}
/* TX power set */

/**@brief Application main function.
 */
int main(void)
{
    // set erase_bonds as true if the security is failed
    bool erase_bonds = true;
    // Initialize.
    log_init();
    timers_init();
    power_management_init();
    ble_stack_init();
    gap_params_init();
    gatt_init();
    services_init();
    // conn_evt_len_ext_set();
    advertising_init();
    conn_params_init();
    tx_power_set();
    /////////////////////
    // Ring buffer init
    ringbuf_init(&uartRx, ringBuffer, UART_RING_SIZE);
    /////////////////////

    // Start execution.
    NRF_LOG_INFO("Device started");
    advertising_start(erase_bonds);

    // Enter main loop.
    for (;;)
    {
        // call this function to perform bonding
        nrf_ble_lesc_request_handler();
        idle_state_handle();
    }
}

/**
 * @}
 */
