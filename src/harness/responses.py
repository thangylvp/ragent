"""Finite Vietnamese responses owned by the demo agent harness.

Every execution-layer outcome maps to a stable key. Those keys are rendered
once, copied to the edge device with a manifest, and looked up at runtime.
Only free-form cloud-agent replies are allowed to use dynamic TTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ResponseTemplate:
    key: str
    text: str


SUCCESS_TEXT = {
    "set_temperature": "Đã điều chỉnh nhiệt độ.",
    "set_fan_speed": "Đã điều chỉnh tốc độ quạt.",
    "set_fan_direction": "Đã điều chỉnh hướng gió.",
    "set_air_recirculation": "Đã điều chỉnh chế độ lấy gió.",
    "set_climate_power": "Đã điều chỉnh hệ thống điều hòa.",
    "control_window": "Đã điều chỉnh cửa sổ.",
    "control_sunroof": "Đã điều chỉnh cửa sổ trời.",
    "control_mirrors": "Đã điều chỉnh gương.",
    "control_trunk": "Đã điều chỉnh cốp xe.",
    "set_kick_sensor": "Đã điều chỉnh cảm biến đá cốp.",
    "set_fog_lights": "Đã điều chỉnh đèn sương mù.",
    "set_auto_high_beam": "Đã điều chỉnh đèn pha tự động.",
    "set_cabin_light": "Đã điều chỉnh đèn trong xe.",
    "set_hud": "Đã điều chỉnh màn hình hắt kính.",
    "set_ambient_light": "Đã điều chỉnh đèn viền.",
    "set_volume": "Đã điều chỉnh âm lượng.",
    "set_regen_braking": "Đã điều chỉnh mức phanh tái sinh.",
    "set_seat_heating": "Đã điều chỉnh sưởi ghế.",
    "set_seat_ventilation": "Đã điều chỉnh thông gió ghế.",
    "set_seat_massage": "Đã điều chỉnh mát xa ghế.",
    "set_screen_brightness": "Đã điều chỉnh độ sáng màn hình.",
    "set_connectivity": "Đã điều chỉnh kết nối.",
    "connect_device": "Đã thực hiện yêu cầu kết nối thiết bị.",
    "set_drive_mode": "Đã chuyển chế độ lái.",
    "set_adas_setting": "Đã điều chỉnh tính năng hỗ trợ lái.",
    "open_app": "Đã mở ứng dụng.",
    "select_media_source": "Đã chọn nguồn phát.",
    "play_media": "Đang phát nội dung bạn yêu cầu.",
    "control_playback": "Đã điều khiển phát nhạc.",
    "control_radio": "Đã điều chỉnh radio.",
    "control_podcast": "Đã điều chỉnh podcast.",
    "set_voice_control": "Đã điều chỉnh điều khiển bằng giọng nói.",
}


TOOL_ACTION = {
    "set_temperature": "điều chỉnh nhiệt độ",
    "set_fan_speed": "điều chỉnh tốc độ quạt",
    "set_fan_direction": "điều chỉnh hướng gió",
    "set_air_recirculation": "đổi chế độ lấy gió",
    "set_climate_power": "điều khiển điều hòa",
    "control_window": "điều khiển cửa sổ",
    "control_sunroof": "điều khiển cửa sổ trời",
    "control_mirrors": "điều khiển gương",
    "control_trunk": "điều khiển cốp xe",
    "set_kick_sensor": "điều khiển cảm biến đá cốp",
    "set_fog_lights": "điều khiển đèn sương mù",
    "set_auto_high_beam": "điều khiển đèn pha tự động",
    "set_cabin_light": "điều khiển đèn trong xe",
    "set_hud": "điều chỉnh màn hình hắt kính",
    "set_ambient_light": "điều chỉnh đèn viền",
    "set_volume": "điều chỉnh âm lượng",
    "set_regen_braking": "điều chỉnh phanh tái sinh",
    "set_seat_heating": "điều chỉnh sưởi ghế",
    "set_seat_ventilation": "điều chỉnh thông gió ghế",
    "set_seat_massage": "điều chỉnh mát xa ghế",
    "set_screen_brightness": "điều chỉnh độ sáng màn hình",
    "set_connectivity": "điều chỉnh kết nối",
    "connect_device": "kết nối thiết bị",
    "set_drive_mode": "đổi chế độ lái",
    "set_adas_setting": "điều chỉnh hỗ trợ lái",
    "open_app": "điều khiển ứng dụng",
    "select_media_source": "chọn nguồn phát",
    "play_media": "phát nội dung",
    "control_playback": "điều khiển phát nhạc",
    "control_radio": "điều khiển radio",
    "control_podcast": "điều khiển podcast",
    "set_voice_control": "điều khiển nhận lệnh bằng giọng nói",
}


MISSING_TEXT = {
    "set_temperature": "Bạn muốn đặt nhiệt độ bao nhiêu, hay tăng giảm bao nhiêu độ?",
    "set_fan_speed": "Bạn muốn đặt quạt ở mức nào, từ không đến bảy?",
    "set_fan_direction": "Bạn muốn gió thổi theo hướng nào?",
    "set_air_recirculation": "Bạn muốn lấy gió trong, gió ngoài hay để tự động?",
    "set_climate_power": "Bạn muốn bật hay tắt điều hòa?",
    "control_window": "Bạn muốn mở hay đóng cửa sổ nào, hoặc đặt cửa ở bao nhiêu phần trăm?",
    "control_sunroof": "Bạn muốn mở, đóng, hé hay nghiêng cửa sổ trời?",
    "control_mirrors": "Bạn muốn gập hay mở gương bên nào?",
    "control_trunk": "Bạn muốn mở hay đóng cốp xe?",
    "set_kick_sensor": "Bạn muốn bật hay tắt cảm biến đá cốp?",
    "set_fog_lights": "Bạn muốn bật hay tắt đèn sương mù?",
    "set_auto_high_beam": "Bạn muốn bật hay tắt đèn pha tự động?",
    "set_cabin_light": "Bạn muốn bật hay tắt đèn trong xe?",
    "set_hud": "Bạn muốn bật tắt, chỉnh độ sáng, hay chỉnh độ cao màn hình hắt kính?",
    "set_ambient_light": "Bạn muốn bật tắt, đổi màu, hay chỉnh độ sáng đèn viền?",
    "set_volume": "Bạn muốn đặt âm lượng bao nhiêu, tăng giảm bao nhiêu, hay tắt tiếng?",
    "set_regen_braking": "Bạn muốn đặt phanh tái sinh ở mức nào?",
    "set_seat_heating": "Bạn muốn sưởi ghế nào và ở mức mấy, từ không đến ba?",
    "set_seat_ventilation": "Bạn muốn thông gió ghế nào và ở mức mấy, từ không đến ba?",
    "set_seat_massage": "Bạn muốn bật tắt, chọn chế độ, hay chỉnh cường độ mát xa cho ghế nào?",
    "set_screen_brightness": "Bạn muốn chỉnh màn hình nào, về bao nhiêu phần trăm, hay chọn chế độ ngày, đêm hoặc tự động?",
    "set_connectivity": "Bạn muốn bật hay tắt kết nối nào, chẳng hạn Wi-Fi hoặc Bluetooth?",
    "connect_device": "Bạn muốn kết nối, ngắt kết nối, ghép đôi hay quên thiết bị nào?",
    "set_drive_mode": "Bạn muốn chuyển sang chế độ lái nào?",
    "set_adas_setting": "Bạn muốn điều chỉnh tính năng hỗ trợ lái nào?",
    "open_app": "Bạn muốn mở hay đóng ứng dụng nào?",
    "select_media_source": "Bạn muốn chọn nguồn phát nào?",
    "play_media": "Bạn muốn nghe bài hát, podcast hoặc nội dung nào?",
    "control_playback": "Bạn muốn phát, tạm dừng, chuyển bài hay thực hiện thao tác nào?",
    "control_radio": "Bạn muốn radio thực hiện thao tác nào? Nếu dò đài, hãy cho biết tần số hoặc tên đài. Nếu dùng kênh nhớ, hãy cho biết số từ một đến hai mươi.",
    "control_podcast": "Bạn muốn podcast thực hiện thao tác nào? Nếu đổi tốc độ, hãy cho biết tốc độ phát.",
    "set_voice_control": "Bạn muốn bật, tắt, đánh thức hay cho hệ thống giọng nói nghỉ?",
}


REJECT_TEXT = {
    "set_temperature": "Không thể điều chỉnh nhiệt độ với giá trị đó. Nhiệt độ hợp lệ là từ 16 đến 30 độ C, hoặc từ 61 đến 86 độ F khi bạn nói rõ đơn vị.",
    "set_fan_speed": "Không thể điều chỉnh quạt với giá trị đó. Tốc độ quạt phải từ mức không đến mức bảy.",
    "set_fan_direction": "Không thể đặt hướng gió đó. Bạn vui lòng chọn một hướng gió được hệ thống hỗ trợ.",
    "set_air_recirculation": "Không thể đặt chế độ lấy gió đó. Các lựa chọn hợp lệ là lấy gió trong, lấy gió ngoài hoặc tự động.",
    "set_climate_power": "Không thể thực hiện lệnh điều hòa đó. Trạng thái phải là bật hoặc tắt.",
    "control_window": "Không thể điều chỉnh cửa sổ với giá trị đó. Vị trí cửa phải từ không đến một trăm phần trăm, hoặc thao tác phải là mở hay đóng.",
    "control_sunroof": "Không thể điều chỉnh cửa sổ trời với giá trị đó. Vị trí phải từ không đến một trăm phần trăm và thao tác phải là mở, đóng, hé hoặc nghiêng.",
    "control_mirrors": "Không thể điều khiển gương với lệnh đó. Bạn có thể gập hoặc mở gương trái, gương phải hay cả hai.",
    "control_trunk": "Không thể điều khiển cốp với lệnh đó. Thao tác hợp lệ là mở hoặc đóng cốp.",
    "set_kick_sensor": "Không thể điều khiển cảm biến đá cốp với lệnh đó. Trạng thái phải là bật hoặc tắt.",
    "set_fog_lights": "Không thể điều khiển đèn sương mù với lệnh đó. Bạn có thể bật hoặc tắt đèn trước, đèn sau hay cả hai.",
    "set_auto_high_beam": "Không thể điều khiển đèn pha tự động với lệnh đó. Trạng thái phải là bật hoặc tắt.",
    "set_cabin_light": "Không thể điều chỉnh đèn trong xe với giá trị đó. Độ sáng phải từ không đến một trăm phần trăm và trạng thái phải là bật hoặc tắt.",
    "set_hud": "Không thể điều chỉnh màn hình hắt kính với giá trị đó. Độ sáng và độ cao phải từ không đến một trăm phần trăm.",
    "set_ambient_light": "Không thể điều chỉnh đèn viền với giá trị đó. Độ sáng phải từ không đến một trăm phần trăm và màu phải thuộc danh sách hệ thống hỗ trợ.",
    "set_volume": "Không thể điều chỉnh âm lượng với giá trị đó. Âm lượng phải từ không đến một trăm phần trăm.",
    "set_regen_braking": "Không thể đặt mức phanh tái sinh đó. Các mức hợp lệ là tắt, thấp, trung bình, cao hoặc lái một bàn đạp.",
    "set_seat_heating": "Không thể đặt mức sưởi ghế đó. Mức hợp lệ là từ không đến ba.",
    "set_seat_ventilation": "Không thể đặt mức thông gió ghế đó. Mức hợp lệ là từ không đến ba.",
    "set_seat_massage": "Không thể điều chỉnh mát xa ghế với giá trị đó. Cường độ phải từ không đến ba và chế độ phải thuộc danh sách được hỗ trợ.",
    "set_screen_brightness": "Không thể điều chỉnh màn hình với giá trị đó. Độ sáng phải từ không đến một trăm phần trăm; chế độ phải là ngày, đêm hoặc tự động.",
    "set_connectivity": "Không thể điều chỉnh kết nối với lệnh đó. Hãy chọn Wi-Fi, Bluetooth, điểm phát sóng, dữ liệu di động hoặc NFC, rồi nói bật hay tắt.",
    "connect_device": "Không thể thực hiện lệnh thiết bị đó. Hệ thống chỉ hỗ trợ Wi-Fi hoặc Bluetooth với thao tác kết nối, ngắt kết nối, ghép đôi hoặc quên thiết bị.",
    "set_drive_mode": "Không thể chọn chế độ lái đó. Bạn vui lòng chọn một chế độ lái được xe hỗ trợ, chẳng hạn tiết kiệm, thoải mái, bình thường hoặc thể thao.",
    "set_adas_setting": "Không thể điều chỉnh hỗ trợ lái với giá trị đó. Hãy chọn đúng tính năng; mức khoảng cách bám xe phải từ một đến năm.",
    "open_app": "Không thể điều khiển ứng dụng với lệnh đó. Hãy cho biết đúng tên ứng dụng và thao tác mở hoặc đóng.",
    "select_media_source": "Không thể chọn nguồn phát đó. Bạn vui lòng chọn một nguồn được hỗ trợ như Bluetooth, USB, radio, podcast hoặc dịch vụ nhạc.",
    "play_media": "Không thể phát nội dung với thông tin đó. Hãy cho biết tên nội dung và, nếu cần, nguồn phát được hỗ trợ.",
    "control_playback": "Không thể điều khiển phát nhạc với lệnh đó. Nếu tua, thời gian phải từ một đến sáu trăm giây; thao tác phải thuộc danh sách được hỗ trợ.",
    "control_radio": "Không thể điều chỉnh radio với giá trị đó. Tần số hợp lệ theo cấu hình hiện tại là từ 76 đến 108; số kênh nhớ phải từ một đến hai mươi.",
    "control_podcast": "Không thể điều khiển podcast với lệnh đó. Tốc độ hợp lệ là 0 phẩy 5, 0 phẩy 75, 1, 1 phẩy 25, 1 phẩy 5 hoặc 2 lần.",
    "set_voice_control": "Không thể điều khiển hệ thống giọng nói với lệnh đó. Trạng thái hợp lệ là bật, tắt, đánh thức hoặc nghỉ.",
}


REPEAT = ResponseTemplate(
    "system.repeat",
    "Xin lỗi, tôi chưa nghe rõ yêu cầu. Bạn vui lòng nói lại nhé.",
)
CLOUD_UNAVAILABLE = ResponseTemplate(
    "system.cloud_unavailable",
    "Xin lỗi, hiện tại tôi chưa kết nối được với trợ lý đám mây. Bạn vui lòng thử lại sau.",
)
TTS_UNAVAILABLE = ResponseTemplate(
    "system.cloud_tts_unavailable",
    "Tôi đã nhận được câu trả lời, nhưng hiện chưa thể phát giọng nói. Bạn vui lòng xem nội dung trên màn hình.",
)
MULTIPLE_CALLS = ResponseTemplate(
    "system.multiple_tool_calls",
    "Tôi chỉ có thể thực hiện một yêu cầu phần cứng trong mỗi lượt. Bạn vui lòng nói từng yêu cầu một.",
)
INVALID_CALL = ResponseTemplate(
    "system.invalid_tool_call",
    "Tôi chưa hiểu chính xác lệnh cần thực hiện. Bạn vui lòng nói lại yêu cầu.",
)
UNKNOWN_TOOL = ResponseTemplate(
    "system.unknown_tool",
    "Tính năng bạn yêu cầu chưa được hệ thống hỗ trợ.",
)
TRUNK_NOT_STOPPED = ResponseTemplate(
    "reject.control_trunk.vehicle_not_stopped",
    "Không thể mở cốp khi xe đang chạy. Bạn hãy dừng xe hoàn toàn và tắt trạng thái vận hành trước khi mở cốp.",
)


class ResponseLibrary:
    def success(self, name: str) -> ResponseTemplate:
        return ResponseTemplate(
            f"success.{name}",
            SUCCESS_TEXT.get(name, "Đã thực hiện yêu cầu của bạn."),
        )

    def missing(self, name: str, fields: Iterable[str] = ()) -> ResponseTemplate:
        del fields  # The per-tool prompt intentionally asks for all actionable data.
        return ResponseTemplate(
            f"missing.{name}",
            MISSING_TEXT.get(name, "Bạn vui lòng cung cấp thêm thông tin cho yêu cầu này."),
        )

    def busy(self, name: str) -> ResponseTemplate:
        action = TOOL_ACTION.get(name, "thực hiện yêu cầu")
        return ResponseTemplate(
            f"busy.{name}",
            f"Hiện tại phần cứng đang bận nên tôi chưa thể {action}. Bạn vui lòng thử lại sau một chút.",
        )

    def reject(self, name: str, reason: str | None = None) -> ResponseTemplate:
        if name == "control_trunk" and reason == "vehicle_not_stopped":
            return TRUNK_NOT_STOPPED
        if not name:
            return INVALID_CALL
        if name not in REJECT_TEXT:
            return UNKNOWN_TOOL
        return ResponseTemplate(f"reject.{name}", REJECT_TEXT[name])

    @property
    def repeat(self) -> ResponseTemplate:
        return REPEAT

    @property
    def cloud_unavailable(self) -> ResponseTemplate:
        return CLOUD_UNAVAILABLE

    @property
    def tts_unavailable(self) -> ResponseTemplate:
        return TTS_UNAVAILABLE

    @property
    def multiple_calls(self) -> ResponseTemplate:
        return MULTIPLE_CALLS

    @property
    def invalid_call(self) -> ResponseTemplate:
        return INVALID_CALL

    def all_templates(self, tool_names: Iterable[str]) -> list[ResponseTemplate]:
        templates = [
            self.repeat,
            self.cloud_unavailable,
            self.tts_unavailable,
            self.multiple_calls,
            self.invalid_call,
            UNKNOWN_TOOL,
        ]
        for name in tool_names:
            if name == "non_tool":
                continue
            templates.extend(
                (
                    self.success(name),
                    self.missing(name),
                    self.busy(name),
                    self.reject(name),
                )
            )
        templates.append(TRUNK_NOT_STOPPED)
        return list({template.key: template for template in templates}.values())
