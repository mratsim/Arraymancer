#!/bin/bash

# Original script from https://github.com/gregvw/amd_sdk/

export OPENCL_VENDOR_PATH=${AMDAPPSDKROOT}/etc/OpenCL/vendors
export LD_LIBRARY_PATH=${AMDAPPSDKROOT}/lib/x86_64:${LD_LIBRARY_PATH}
export CMAKE_LIBRARY_PATH=${AMDAPPSDKROOT}/lib/x86_64

if [ ! -e ${AMDAPPSDKROOT}/bin/x86_64/clinfo ]; then
    # Location from which get nonce and file name from
    URL="https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/"
    URLDOWN="https://developer.amd.com/amd-license-agreement-appsdk/"

    NONCE1_STRING='name="amd_developer_central_downloads_page_nonce"'
    FILE_STRING='name="f"'
    POSTID_STRING='name="post_id"'
    NONCE2_STRING='name="amd_developer_central_nonce"'

    # This gets the second latest (2.9.1 ATM, latest is 3.0)
    # For newest: FORM=`wget -qO - $URL | sed -n '/download-2/,/64-bit/p'`
    FORM=`wget -qO - $URL | sed -n '/download-5/,/64-bit/p'`

    # Get nonce from form
    NONCE1=`echo $FORM | awk -F ${NONCE1_STRING} '{print $2}'`
    NONCE1=`echo $NONCE1 | awk -F'"' '{print $2}'`
    echo $NONCE1

    # get the postid
    POSTID=`echo $FORM | awk -F ${POSTID_STRING} '{print $2}'`
    POSTID=`echo $POSTID | awk -F'"' '{print $2}'`
    echo $POSTID

    # get file name
    FILE=`echo $FORM | awk -F ${FILE_STRING} '{print $2}'`
    FILE=`echo $FILE | awk -F'"' '{print $2}'`
    echo $FILE

    FORM=`wget -qO - $URLDOWN --post-data "amd_developer_central_downloads_page_nonce=${NONCE1}&f=${FILE}&post_id=${POSTID}"`

    NONCE2=`echo $FORM | awk -F ${NONCE2_STRING} '{print $2}'`
    NONCE2=`echo $NONCE2 | awk -F'"' '{print $2}'`
    echo $NONCE2

    wget --content-disposition --trust-server-names $URLDOWN --post-data "amd_developer_central_nonce=${NONCE2}&f=${FILE}" -O AMD-SDK.tar.bz2;

    # Unpack and install
    tar -xjf AMD-SDK.tar.bz2;
    mkdir -p ${OPENCL_VENDOR_PATH};
    sh AMD-APP-SDK*.sh --tar -xf -C ${AMDAPPSDKROOT};
    echo libamdocl64.so > ${OPENCL_VENDOR_PATH}/amdocl64.icd;
    chmod +x ${AMDAPPSDKROOT}/bin/x86_64/clinfo;
fi

${AMDAPPSDKROOT}/bin/x86_64/clinfo