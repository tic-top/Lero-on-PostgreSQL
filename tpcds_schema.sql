CREATE TABLE call_center (
    cc_call_center_sk         INTEGER        NOT NULL,
    cc_call_center_id         CHAR(16)       NOT NULL,
    cc_rec_start_date         DATE           NOT NULL,
    cc_rec_end_date           DATE,
    cc_closed_date_sk         INTEGER,
    cc_open_date_sk           INTEGER        NOT NULL,
    cc_name                   VARCHAR(50)    NOT NULL,
    cc_class                  VARCHAR(50)    NOT NULL,
    cc_employees              INTEGER        NOT NULL,
    cc_sq_ft                  INTEGER,
    cc_hours                  CHAR(20),
    cc_manager                VARCHAR(40),
    cc_mkt_id                 INTEGER        NOT NULL,
    cc_mkt_class              CHAR(50),
    cc_mkt_desc               VARCHAR(200),
    cc_market_manager         VARCHAR(40),
    cc_division               INTEGER        NOT NULL,
    cc_division_name          VARCHAR(50),
    cc_company                INTEGER        NOT NULL,
    cc_company_name           CHAR(50),
    cc_street_number          CHAR(10),
    cc_street_name            VARCHAR(60),
    cc_street_type            CHAR(15),
    cc_suite_number           CHAR(10),
    cc_city                   VARCHAR(60),
    cc_county                 VARCHAR(30),
    cc_state                  CHAR(2),
    cc_zip                    CHAR(10),
    cc_country                VARCHAR(20),
    cc_gmt_offset             DECIMAL(5,2)   NOT NULL,
    cc_tax_percentage         DECIMAL(5,2)   NOT NULL
);

CREATE TABLE catalog_page (
    cp_catalog_page_sk        integer               not null,
    cp_catalog_page_id        char(16)              not null,
    cp_start_date_sk          integer,
    cp_end_date_sk            integer,
    cp_department             varchar(50),
    cp_catalog_number         integer,
    cp_catalog_page_number    integer,
    cp_description            varchar(100),
    cp_type                   varchar(100),
    PRIMARY KEY (cp_catalog_page_sk)
);

CREATE TABLE catalog_returns (
    cr_returned_date_sk       integer,
    cr_returned_time_sk       integer,
    cr_item_sk                integer               not null,
    cr_refunded_customer_sk   integer,
    cr_refunded_cdemo_sk      integer,
    cr_refunded_hdemo_sk      integer,
    cr_refunded_addr_sk       integer,
    cr_returning_customer_sk  integer,
    cr_returning_cdemo_sk     integer,
    cr_returning_hdemo_sk     integer,
    cr_returning_addr_sk      integer,
    cr_call_center_sk         integer,
    cr_catalog_page_sk        integer,
    cr_ship_mode_sk           integer,
    cr_warehouse_sk           integer,
    cr_reason_sk              integer,
    cr_order_number           integer               not null,
    cr_return_quantity        integer,
    cr_return_amount          decimal(7,2),
    cr_return_tax             decimal(7,2),
    cr_return_amt_inc_tax     decimal(7,2),
    cr_fee                    decimal(7,2),
    cr_return_ship_cost       decimal(7,2),
    cr_refunded_cash          decimal(7,2),
    cr_reversed_charge        decimal(7,2),
    cr_store_credit           decimal(7,2),
    cr_net_loss               decimal(7,2),
    PRIMARY KEY (cr_item_sk, cr_order_number)
);

CREATE TABLE customer (
    c_customer_sk             integer               not null,
    c_customer_id             char(16)              not null,
    c_current_cdemo_sk        integer,
    c_current_hdemo_sk        integer,
    c_current_addr_sk         integer,
    c_first_shipto_date_sk    integer,
    c_first_sales_date_sk     integer,
    c_salutation              char(10),
    c_first_name              char(20),
    c_last_name               char(30),
    c_preferred_cust_flag     char(1),
    c_birth_day               integer,
    c_birth_month             integer,
    c_birth_year              integer,
    c_birth_country           varchar(20),
    c_login                   char(13),
    c_email_address           char(50),
    c_last_review_date        char(10),
    PRIMARY KEY (c_customer_sk)
);

CREATE TABLE customer_address (
    ca_address_sk             integer               not null,
    ca_address_id             char(16)              not null,
    ca_street_number          char(10),
    ca_street_name            varchar(60),
    ca_street_type            char(15),
    ca_suite_number           char(10),
    ca_city                   varchar(60),
    ca_county                 varchar(30),
    ca_state                  char(2),
    ca_zip                    char(10),
    ca_country                varchar(20),
    ca_gmt_offset             decimal(5,2),
    ca_location_type          char(20),
    PRIMARY KEY (ca_address_sk)
);

CREATE TABLE customer_demographics (
    cd_demo_sk                integer               not null,
    cd_gender                 char(1),
    cd_marital_status         char(1),
    cd_education_status       char(20),
    cd_purchase_estimate      integer,
    cd_credit_rating          char(10),
    cd_dep_count              integer,
    cd_dep_employed_count     integer,
    cd_dep_college_count      integer,
    PRIMARY KEY (cd_demo_sk)
);

CREATE TABLE date_dim (
    d_date_sk                 integer               not null,
    d_date_id                 char(16)              not null,
    d_date                    date,
    d_month_seq               integer,
    d_week_seq                integer,
    d_quarter_seq             integer,
    d_year                    integer,
    d_dow                     integer,
    d_moy                     integer,
    d_dom                     integer,
    d_qoy                     integer,
    d_fy_year                 integer,
    d_fy_quarter_seq          integer,
    d_fy_week_seq             integer,
    d_day_name                char(9),
    d_quarter_name            char(6),
    d_holiday                 char(1),
    d_weekend                 char(1),
    d_following_holiday       char(1),
    d_first_dom               integer,
    d_last_dom                integer,
    d_same_day_ly             integer,
    d_same_day_lq             integer,
    d_current_day             char(1),
    d_current_week            char(1),
    d_current_month           char(1),
    d_current_quarter         char(1),
    d_current_year            char(1),
    PRIMARY KEY (d_date_sk)
);

CREATE TABLE dbgen_version (
    dv_version                varchar(16),
    dv_create_date            date,
    dv_create_time            time,
    dv_cmdline_args           varchar(200)
);

CREATE TABLE household_demographics (
    hd_demo_sk                integer               not null,
    hd_income_band_sk         integer,
    hd_buy_potential          char(15),
    hd_dep_count              integer,
    hd_vehicle_count          integer,
    PRIMARY KEY (hd_demo_sk)
);

CREATE TABLE income_band (
    ib_income_band_sk         integer               not null,
    ib_lower_bound            integer,
    ib_upper_bound            integer,
    PRIMARY KEY (ib_income_band_sk)
);

CREATE TABLE item (
    i_item_sk                 integer               not null,
    i_item_id                 char(16)              not null,
    i_rec_start_date          date,
    i_rec_end_date            date,
    i_item_desc               varchar(200),
    i_current_price           decimal(7,2),
    i_wholesale_cost          decimal(7,2),
    i_brand_id                integer,
    i_brand                   char(50),
    i_class_id                integer,
    i_class                   char(50),
    i_category_id             integer,
    i_category                char(50),
    i_manufact_id             integer,
    i_manufact                char(50),
    i_size                    char(20),
    i_formulation             char(20),
    i_color                   char(20),
    i_units                   char(10),
    i_container               char(10),
    i_manager_id              integer,
    i_product_name            char(50),
    PRIMARY KEY (i_item_sk)
);

CREATE TABLE promotion (
    p_promo_sk                integer               not null,
    p_promo_id                char(16)              not null,
    p_start_date_sk           integer,
    p_end_date_sk             integer,
    p_item_sk                 integer,
    p_cost                    decimal(15,2),
    p_response_target         integer,
    p_promo_name              char(50),
    p_channel_dmail           char(1),
    p_channel_email           char(1),
    p_channel_catalog         char(1),
    p_channel_tv              char(1),
    p_channel_radio           char(1),
    p_channel_press           char(1),
    p_channel_event           char(1),
    p_channel_demo            char(1),
    p_channel_details         varchar(100),
    p_purpose                 char(15),
    p_discount_active         char(1),
    PRIMARY KEY (p_promo_sk)
);

CREATE TABLE reason (
    r_reason_sk               integer               not null,
    r_reason_id               char(16)              not null,
    r_reason_desc             char(100),
    PRIMARY KEY (r_reason_sk)
);

CREATE TABLE ship_mode (
    sm_ship_mode_sk           integer               not null,
    sm_ship_mode_id           char(16)              not null,
    sm_type                   char(30),
    sm_code                   char(10),
    sm_carrier                char(20),
    sm_contract               char(20),
    PRIMARY KEY (sm_ship_mode_sk)
);

CREATE TABLE store (
    s_store_sk                integer               not null,
    s_store_id                char(16)              not null,
    s_rec_start_date          date,
    s_rec_end_date            date,
    s_closed_date_sk          integer,
    s_store_name              varchar(50),
    s_number_employees        integer,
    s_floor_space            integer,
    s_hours                   char(20),
    s_manager                 varchar(40),
    s_market_id               integer,
    s_geography_class         varchar(100),
    s_market_desc             varchar(100),
    s_market_manager          varchar(40),
    s_division_id             integer,
    s_division_name           varchar(50),
    s_company_id              integer,
    s_company_name            varchar(50),
    s_street_number           varchar(10),
    s_street_name             varchar(60),
    s_street_type             char(15),
    s_suite_number            char(10),
    s_city                    varchar(60),
    s_county                  varchar(30),
    s_state                   char(2),
    s_zip                     char(10),
    s_country                 varchar(20),
    s_gmt_offset              decimal(5,2),
    s_tax_precentage          decimal(5,2),
    PRIMARY KEY (s_store_sk)
);

CREATE TABLE store_returns (
    sr_returned_date_sk       integer,
    sr_return_time_sk         integer,
    sr_item_sk                integer               not null,
    sr_customer_sk            integer,
    sr_cdemo_sk               integer,
    sr_hdemo_sk               integer,
    sr_addr_sk                integer,
    sr_store_sk               integer,
    sr_reason_sk              integer,
    sr_ticket_number          integer               not null,
    sr_return_quantity        integer,
    sr_return_amt             decimal(7,2),
    sr_return_tax             decimal(7,2),
    sr_return_amt_inc_tax     decimal(7,2),
    sr_fee                    decimal(7,2),
    sr_return_ship_cost       decimal(7,2),
    sr_refunded_cash          decimal(7,2),
    sr_reversed_charge        decimal(7,2),
    sr_store_credit           decimal(7,2),
    sr_net_loss               decimal(7,2),
    PRIMARY KEY (sr_item_sk, sr_ticket_number)
);

CREATE TABLE time_dim (
    t_time_sk                 integer               not null,
    t_time_id                 char(16)              not null,
    t_time                    integer,
    t_hour                    integer,
    t_minute                  integer,
    t_second                  integer,
    t_am_pm                   char(2),
    t_shift                   char(20),
    t_sub_shift               char(20),
    t_meal_time               char(20),
    PRIMARY KEY (t_time_sk)
);

CREATE TABLE warehouse (
    w_warehouse_sk            integer               not null,
    w_warehouse_id            char(16)              not null,
    w_warehouse_name          varchar(20),
    w_warehouse_sq_ft         integer,
    w_street_number           char(10),
    w_street_name             varchar(60),
    w_street_type             char(15),
    w_suite_number            char(10),
    w_city                    varchar(60),
    w_county                  varchar(30),
    w_state                   char(2),
    w_zip                     char(10),
    w_country                 varchar(20),
    w_gmt_offset              decimal(5,2),
    PRIMARY KEY (w_warehouse_sk)
);

CREATE TABLE web_page (
    wp_web_page_sk            integer               not null,
    wp_web_page_id            char(16)              not null,
    wp_rec_start_date         date,
    wp_rec_end_date           date,
    wp_creation_date_sk       integer,
    wp_access_date_sk         integer,
    wp_autogen_flag           char(1),
    wp_customer_sk            integer,
    wp_url                    varchar(100),
    wp_type                   char(50),
    wp_char_count             integer,
    wp_link_count             integer,
    wp_image_count            integer,
    wp_max_ad_count           integer,
    PRIMARY KEY (wp_web_page_sk)
);

CREATE TABLE web_returns (
    wr_returned_date_sk       integer,
    wr_returned_time_sk       integer,
    wr_item_sk                integer               not null,
    wr_refunded_customer_sk   integer,
    wr_refunded_cdemo_sk      integer,
    wr_refunded_hdemo_sk      integer,
    wr_refunded_addr_sk       integer,
    wr_returning_customer_sk  integer,
    wr_returning_cdemo_sk     integer,
    wr_returning_hdemo_sk     integer,
    wr_returning_addr_sk      integer,
    wr_web_page_sk            integer,
    wr_reason_sk              integer,
    wr_order_number           integer               not null,
    wr_return_quantity        integer,
    wr_return_amt             decimal(7,2),
    wr_return_tax             decimal(7,2),
    wr_return_amt_inc_tax     decimal(7,2),
    wr_fee                    decimal(7,2),
    wr_return_ship_cost       decimal(7,2),
    wr_refunded_cash          decimal(7,2),
    wr_reversed_charge        decimal(7,2),
    wr_account_credit         decimal(7,2),
    wr_net_loss               decimal(7,2),
    PRIMARY KEY (wr_item_sk, wr_order_number)
);

CREATE TABLE web_site (
    web_site_sk               integer               not null,
    web_site_id               char(16)              not null,
    web_rec_start_date        date,
    web_rec_end_date          date,
    web_name                  varchar(50),
    web_open_date_sk          integer,
    web_close_date_sk         integer,
    web_class                 varchar(50),
    web_manager               varchar(40),
    web_mkt_id                integer,
    web_mkt_class             varchar(50),
    web_mkt_desc              varchar(100),
    web_market_manager        varchar(40),
    web_company_id            integer,
    web_company_name          char(50),
    web_street_number         char(10),
    web_street_name           varchar(60),
    web_street_type           char(15),
    web_suite_number          char(10),
    web_city                  varchar(60),
    web_county                varchar(30),
    web_state                 char(2),
    web_zip                   char(10),
    web_country               varchar(20),
    web_gmt_offset            decimal(5,2),
    web_tax_percentage        decimal(5,2),
    PRIMARY KEY (web_site_sk)
);