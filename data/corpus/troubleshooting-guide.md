# Troubleshooting Guide & Error Codes (KB-TS-06)

## Connectivity issues

### Bulb or plug won't connect during setup (error LL-E10 "Pairing timeout")
1. Confirm your phone is on the 2.4 GHz band — devices cannot join 5 GHz-only networks.
2. Disable VPN on your phone during setup.
3. Move the device within 3 meters of the router for pairing.
4. Factory reset (bulbs: power-cycle 3 times; plug: hold side button 10 seconds) and retry.

### Device shows "Offline" in app (error LL-E20)
Usually a Wi-Fi issue. Power-cycle the device and your router. If only one device is offline while others work, re-run setup for that device. Devices reconnect automatically after a power outage within 2 minutes.

### Bulb flickers (error LL-E12 "Dimmer interference")
LumenLeaf bulbs are incompatible with wall dimmer switches, including "smart compatible" TRIAC dimmers. Replace the dimmer with a standard on/off switch. Flicker with a standard switch indicates a defective bulb — covered by warranty.

### Hub blinking red (error LL-E30 "No internet")
Check the Ethernet cable, then your ISP connection. The hub requires outbound port 443 and 8883; some corporate or guest networks block MQTT (port 8883), which prevents the hub from reaching our cloud.

## App issues

### App can't find device although it's powered (error LL-E11)
Enable Bluetooth and Local Network permission for the LumenLeaf app (iOS Settings → Privacy). Android requires Location permission for Wi-Fi scanning.

### Scenes lag or run out of order
Without a hub, scenes execute via cloud and can take 2–3 seconds. Add an LL-H300 hub for local execution under 300 ms.

## Overload on smart plug (error LL-E45)
The LL-P50 cut power because the connected load exceeded 1800W. Unplug the appliance, press the side button once to clear, and connect a smaller load. Repeated LL-E45 events with loads under 1800W indicate a faulty unit — contact support for warranty replacement.

## When to escalate to a human agent
Contact a support agent if: the device is physically damaged on arrival, you smell burning or see scorch marks (stop using immediately), an error persists after factory reset, or your issue involves a payment dispute. Reach us via live chat (24/7) or support@lumenleaf.example.
